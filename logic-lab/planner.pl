connected(a,b).
connected(b,a).
connected(b,c).
connected(c,b).
connected(d,e).

can_move(X,Y) :-
    connected(X,Y).

valid_move(X,Y) :-
    can_reach(X, Y, [X]).



can_reach(X, Y, _) :-
    connected(X, Y).

can_reach(X, Y, Visited) :-
    connected(X, Z),
    \+ member(Z, Visited),
    can_reach(Z, Y, [Z | Visited]).


wet_road.
slippery :-
    wet_road.
reduce_speed :-
    slippery