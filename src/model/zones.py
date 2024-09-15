from copy import copy

import torch


class Zone:
    """
    A Zone is represented with physical coordinates and "virtual" coordinates. Physical coordinates
    dictate how to map from a real pitch to one of our zones. Virtual coordinates are used to place the zone
    in our one-hot encodings.

    See Zones for the full context.
    """

    def __init__(self, coords: list[tuple[int, int]], left: float, right: float, bottom: float, top: float, is_strike: bool = True, is_borderline: bool = False):
        self.coords = coords
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.is_strike = is_strike
        self.is_borderline = is_borderline

    def center(self) -> tuple[float, float]:
        """Returns the center of the zone."""

        return (self.left + self.right) / 2, (self.bottom + self.top) / 2

    def __repr__(self):
        return f"Zone({self.coords}){' borderline' if self.is_borderline else ''}{' strike' if self.is_strike else ''}"

    def __hash__(self):
        return hash((*self.coords, self.is_strike, self.is_borderline))

    def __eq__(self, other):
        return hash(self) == hash(other)


class Zones:
    """
    We use the following reference for default measurements https://tangotiger.net/strikezone/zone%20chart.png

    Using those measurements, we divide the physical strike zone into 5x5 zones. In our model, (0, 0)
    is the bottom left corner (although how you orient it doesn't actually matter).

    * ----- *
    | o o o |
    | o o o |
    | o o o |
    * ----- *

    Borderline zones are also included. These are pitches that land in the "shadow zone", one baseball's width around
    the edge of the strike zone (within and without). We use this to handle batter patience.

    In our computations, zone indexes are used instead of the direct Zone objects. A singleton instance of Zones (default)
    makes it easy to fetch a Zone object from an index.

    COMBINED_ZONES is a list of ZONES and BORDERLINE_ZONES
    The only difference between a borderline zone and a regular zone in the actual data is the is_borderline attribute,
    meaning that an index of a borderline zone is the index of the corresponding zone in index ZONES + len(ZONES)
    """

    # The width/height of the virtual strike zone. This probably cannot be changed without breaking things.
    DIMENSION = 5

    def __init__(self, width=20, sz_top=42, sz_bottom=18):
        """
        :param width: The width of the strike zone
        :param sz_top: The top of the strike zone (a batter's shoulders)
        :param sz_bottom: The bottom of the strike zone (a batter's knees)
        """

        self.ZONES = []

        # Strike zones
        strike_zone_dim = 3
        strike_left = -width / 2
        strike_height = sz_top - sz_bottom
        x_step = width / strike_zone_dim
        y_step = strike_height / strike_zone_dim
        self.ZONES.extend([
            Zone([(i + 1, j + 1)],
                 strike_left + i * x_step, strike_left + (i + 1) * x_step,
                 sz_bottom + j * y_step, sz_bottom + (j + 1) * y_step, is_strike=True)
            for i in range(strike_zone_dim) for j in range(strike_zone_dim)
        ])

        # Ball zones
        inf = float('inf')
        self.ZONES.extend([
            Zone([(0, 0)], -inf, strike_left, -inf, sz_bottom, is_strike=False),
            Zone([(0, 4)], -inf, strike_left, sz_top, inf, is_strike=False),
            Zone([(4, 0)], strike_left + width, inf, -inf, sz_bottom, is_strike=False),
            Zone([(4, 4)], strike_left + width, inf, sz_top, inf, is_strike=False),

            Zone([(0, 1), (0, 2), (0, 3)], -inf, strike_left, sz_bottom, sz_top, is_strike=False),
            Zone([(4, 1), (4, 2), (4, 3)], strike_left + width, inf, sz_bottom, sz_top, is_strike=False),
            Zone([(1, 0), (2, 0), (3, 0)], strike_left, strike_left + width, -inf, sz_bottom, is_strike=False),
            Zone([(1, 4), (2, 4), (3, 4)], strike_left, strike_left + width, sz_top, inf, is_strike=False)
        ])

        self.BORDERLINE_ZONES = [copy(zone) for zone in self.ZONES]
        for zone in self.BORDERLINE_ZONES:
            zone.is_borderline = True

        self.COMBINED_ZONES = self.ZONES + self.BORDERLINE_ZONES

        self.STRIKE_ZONE_WIDTH = width
        self.STRIKE_ZONE_HEIGHT = sz_top - sz_bottom
        self.STRIKE_ZONE_BOTTOM = sz_bottom
        self.STRIKE_ZONE_TOP = sz_top
        self.STRIKE_ZONE_LEFT = strike_left
        self.STRIKE_ZONE_RIGHT = strike_left + width
        self.BALL_SIZE = 3

    def get_zone(self, x_loc: float | None, y_loc: float | None) -> tuple[int, Zone] | None:
        """Converts physical coordinates x and y (in inches) to virtual coordinates in the strike zone."""

        if x_loc is None or y_loc is None:
            return None

        for i, zone in enumerate(self.ZONES):
            if zone.left <= x_loc <= zone.right and zone.bottom <= y_loc <= zone.top:
                if self.STRIKE_ZONE_LEFT - self.BALL_SIZE <= x_loc <= self.STRIKE_ZONE_RIGHT + self.BALL_SIZE and \
                    self.STRIKE_ZONE_BOTTOM - self.BALL_SIZE <= y_loc <= self.STRIKE_ZONE_TOP + self.BALL_SIZE and \
                    not (self.STRIKE_ZONE_LEFT + self.BALL_SIZE <= x_loc <= self.STRIKE_ZONE_RIGHT - self.BALL_SIZE and
                         self.STRIKE_ZONE_BOTTOM + self.BALL_SIZE <= y_loc <= self.STRIKE_ZONE_TOP - self.BALL_SIZE):
                    return i + len(self.ZONES), self.BORDERLINE_ZONES[i]
                else:
                    return i, zone

    def get_zones_batched(self, x_locs: torch.Tensor, y_locs: torch.Tensor) -> list[int]:
        """
        This batched version of get_zone is much faster and necessary for the random sampling method used for
        generating a pitch outcome distribution. It returns indices.
        """

        borderline_mask = (self.STRIKE_ZONE_LEFT - self.BALL_SIZE <= x_locs) & (x_locs <= self.STRIKE_ZONE_RIGHT + self.BALL_SIZE) & \
                          (self.STRIKE_ZONE_BOTTOM - self.BALL_SIZE <= y_locs) & (y_locs <= self.STRIKE_ZONE_TOP + self.BALL_SIZE) & \
                          ~((self.STRIKE_ZONE_LEFT + self.BALL_SIZE <= x_locs) & (x_locs <= self.STRIKE_ZONE_RIGHT - self.BALL_SIZE) &
                            (self.STRIKE_ZONE_BOTTOM + self.BALL_SIZE <= y_locs) & (y_locs <= self.STRIKE_ZONE_TOP - self.BALL_SIZE))

        result_zones: list = [-1] * len(x_locs)
        for zone_i, zone in enumerate(self.ZONES):
            mask = (zone.left <= x_locs) & (x_locs <= zone.right) & (zone.bottom <= y_locs) & (y_locs <= zone.top)
            indices = torch.nonzero(mask, as_tuple=False).squeeze()
            if indices.numel() == 1:
                result_zones[indices.item()] = zone_i + len(self.ZONES) * borderline_mask[indices.item()]
            elif indices.numel() > 1:
                for idx in indices:
                    result_zones[idx.item()] = zone_i + len(self.ZONES) * borderline_mask[idx.item()]

        return result_zones


# For convenience, here is a default instance of Zones
default = Zones()
