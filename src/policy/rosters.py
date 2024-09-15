'''
For convenience, here are some lists of batting lineups for a bunch of MLB teams. We should have enough
data for each of these players.

There are also some pitchers in the pitchers dictionary.
'''

rosters = {
    # Masyn Winn, Alec Burleson, Willson Contreras, Paul Goldschmidt, Brendan Donovan, Nolan Arenado, Lars Nootbaar, Matt Carpenter, Nolan Gorman
    'cardinals': [691026, 676475, 575929, 502671, 680977, 571448, 663457, 572761, 669357],

    # Nico Hoerner, Michael Busch, Seiya Suzuki, Ian Happ, Christopher Morel, Dansby Swanson, Miles Mastrobuoni, David Bote, Tomas Nido
    'cubs': [663538, 683737, 673548, 664023, 666624, 621020, 670156, 623520, 621512],

    # Adam Frazier, Bobby Witt Jr., Vinnie Pasquantino, Salvador Perez, Michael Massey, Hunter Renfroe, MJ Melendez, Freddy Fermin, Kyle Isbel
    'royals': [624428, 677951, 686469, 521692, 686681, 592669, 669004, 666023, 664728],

    # Jose Altuve, Jeremy Peña, Yordan Alvarez, Jake Meyers, Jon Singleton, Yainer Diaz, Trey Cabbage, Jose Abreu, Mauricio Dubón
    'astros': [514888, 665161, 670541, 676694, 572138, 673237, 663550, 547989, 643289],

    # Luis García Jr., Joey Meneses, Dominic Smith, Jeimer Candelario, Corey Dickerson, Keibert Ruiz, Alex Call, Ildemaro Vargas, Lane Thomas
    'nationals': [671277, 608841, 642086, 600869, 572816, 660688, 669743, 545121, 657041],

    # Anthony Volpe, Gleyber Torres, Anthony Rizzo, Giancarlo Stanton, Oswaldo Cabrera, Aaron Hicks, Oswald Peraza, Jose Trevino, Willie Calhoun
    'yankees': [683011, 650402, 519203, 519317, 665828, 543305, 672724, 624431, 641432],

    # Luis Arraez, Jorge Soler, Garrett Cooper, Avisail Garcia, Jean Segura, Jesus Sanchez, Bryan De La Cruz, Joey Wendle, Jacob Stallings
    'marlins': [650333, 624585, 643265, 541645, 516416, 660821, 650559, 621563, 607732],

    # Brandon Nimmo, Starling Marte, Mark Canha, Darin Ruf, Tommy Pham, Tomas Nido, Mark Vientos, Ronny Mauricio, Jose Peraza
    'mets': [607043, 516782, 592192, 573131, 502054, 621512, 668901, 677595, 606299],

    # Ke'Bryan Hayes, Bryan Reynolds, Michael Chavis, Ben Gamel, Diego Castillo, Yoshi Tsutsugo, Rodolfo Castro, Michael Pérez, Jack Suwinski
    'pirates': [663647, 668804, 656308, 592325, 660636, 660294, 666801, 605421, 669261],

    # Yoan Moncada, Tim Anderson, Jose Abreu, Matt Davidson, Nicky Delmonico, Welington Castillo, Trayce Thompson, Adam Engel, James Shields
    'white sox': [660162, 641313, 547989, 571602, 547170, 456078, 572204, 641553, 448306],

    # Joe Mauer, Brian Dozier, Max Kepler, Eduardo Escobar, Eddie Rosario, Robbie Grossman, Bobby Wilson, Ehire Adrianza, Christian Vazquez
    'twins': [408045, 572821, 596146, 500871, 592696, 543257, 435064, 501303, 543877],

    # Cesar Hernandez, Rhys Hoskins, Odubel Herrera, Carlos Santana, Scott Kingery, Nick Williams, Maikel Franco, Jorge Alfaro, Jake Arrieta
    'phillies': [514917, 656555, 546318, 467793, 664068, 608384, 596748, 595751, 453562],

    # Eric Thames, Christian Yelich, Lorenzo Cain, Travis Shaw, Jesus Aguilar, Brad Miller, Hernan Perez, Manny Piña, Chase Anderson
    'brewers': [519346, 592885, 456715, 543768, 542583, 543543, 541650, 444489, 502624],

    # Francisco Lindor, Michael Brantley, Jose Ramirez, Yonder Alonso, Lonnie Chisenhall, Jason Kipnis, Yan Gomes, Tyler Naquin, Brayan Rocchio
    'guardians': [596019, 488726, 608070, 475174, 502082, 543401, 543228, 571980, 677587],

    # Ender Inciarte, Ozzie Albies, Freddie Freeman, Nick Markakis, Tyler Flowers, Ronald Acuña Jr., Johan Camargo, Dansby Swanson, Julio Teheran
    'braves': [542255, 645277, 518692, 455976, 452095, 660670, 622666, 621020, 527054],

    # Alen Hanson, Buster Posey, Andrew McCutchen, Brandon Belt, Brandon Crawford, Pablo Sandoval, Joe Panik, Gorkys Hernandez, Dereck Rodriguez
    'giants': [593700, 457763, 457705, 474832, 543063, 467055, 605412, 491676, 605446],

    # Charlie Blackmon, Ian Desmond, Nolan Arenado, Carlos González, Ryan McMahon, Tom Murphy, Gerardo Parra, Garrett Hampson, Tyler Anderson
    'rockies': [453568, 435622, 571448, 471865, 641857, 608596, 467827, 641658, 542881],

    # Ian Kinsler, Eric Hosmer, Manny Machado, Hunter Renfroe, Franmil Reyes, Fernando Tatis Jr., Austin Hedges, Manuel Margot, Kyle Higashioka
    'padres': [435079, 543333, 592518, 592669, 614177, 665487, 595978, 622534, 543309],

    # Joc Pederson, Corey Seager, Cody Bellinger, A.J. Pollock, Max Muncy, Alex Verdugo, Enrique Hernández, Russell Martin, Hyun Jin Ryu
    'dodgers': [592626, 608369, 641355, 572041, 571970, 657077, 571771, 431145, 547943],

    # Joey Votto, Eugenio Suárez, Jesse Winker, Yasiel Puig, Tucker Barnhart, Jose Iglesias, Scott Schebler, Sonny Gray, Jose Peraza
    'reds': [458015, 553993, 608385, 624577, 571466, 578428, 594988, 543243, 606299]
}

pitchers = {
    # Sonny Gray
    'cardinals': 543243,

    # Javier Assad
    'cubs': 665871,

    # Hunter Harvey
    'royals': 640451,

    # Framber Valdez
    'astros': 664285,

    # Patrick Corbin
    'nationals': 571578,

    # Gerrit Cole
    'yankees': 543037,

    # Yonny Chirinos
    'marlins': 630023,

    # Sean Manaea
    'mets': 640455,

    # Marco Gonzales
    'pirates': 594835,

    # Erick Fedde
    'white sox': 607200,

    # Pablo Lopez
    'twins': 641154,

    # Zack Wheeler
    'phillies': 554430,

    # Freddy Peralta
    'brewers': 642547,

    # Tanner Bibee
    'guardians': 676440,

    # Chris Sales
    'braves': 519242,

    # Logan Webb
    'giants': 657277,

    # Cal Quantrill
    'rockies': 615698,

    # Dylan Cease
    'padres': 656302,

    # Tyler Glasnow
    'dodgers': 607192,

    # Frankie Montas
    'reds': 593423
}