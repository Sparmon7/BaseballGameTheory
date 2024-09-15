## Limitations
- Our model is currently oblivious to plays that happen on field, besides the number of bases achieved
by the runner and whether the batter gets out. This is a significant part of the game that we are ignoring. ↓↓

- The pitcher's options are limited in a way that might not be representative of the actual game. The pitcher cannot
aim for a borderline zone and can only aim for 5x5 centers.

- Our player representations definitely do not include all relevant information, particularly when it comes
  to lineup optimization. Another thing is we might want to weight more recent games more heavily.

## TODO / Ideas
- Investigate the problem of player trade value, with respect to likely pitchers and lineups to be encountered in playoffs.

- Incorporate a distribution for on-field outcomes. We can start with the empirical distribution

- Is there any way to approximate values for the full game using the simplified game rules? This would allow
  for more complex analysis. Perhaps viewing the states as vectors would help? Transforming from one representation to another...

- I'm thinking a modified greedy hill climbing might work better. Instead of the swap operation, what if we do an
  insert/move? Like when you move a card from one location to another