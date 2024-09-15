// Run the following in the console of https://www.spotrac.com/mlb/transactions/trade/_/start/2008-01-01/end/2024-12-31
// to scrape the transaction data. Copy the output object and paste it into raw_data/transactions.json

let data = [];

const trades = document.querySelectorAll(".tradetable > tbody");

trades.forEach(trade => {
    const date = trade.children[0].children[0].children[0].children[0].innerHTML;

    const getTeamData = (teamContainer) => {
        try {
            return {
                name: teamContainer.children[0].children[0].children[0].alt,
                acquires: Array.from(teamContainer.children[0].children[1].children[0].querySelectorAll(".tradeplayer"))
                    .map(player => {
                        if (player.children.length > 0) {
                            return player.children[0].innerHTML.replace(/\s*\(.*?\)/g, "");
                        } else {
                            return player.innerHTML;
                        }
                    })
            };
        } catch (_) {}
    }

    const team1 = getTeamData(trade.children[1].children[0].children[0].children[0])
    const team2 = getTeamData(trade.children[1].children[0].children[0].children[1])

    data.push({
        date: date,
        team1: team1,
        team2: team2
    });
});

console.log(data);