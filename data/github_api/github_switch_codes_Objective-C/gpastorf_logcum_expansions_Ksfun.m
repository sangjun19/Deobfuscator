// Repository: gpastorf/logcum_expansions
// File: Ksfun.m

function KS = Ksfun(order)

switch order,
    case 1, KS = 1;
    case 2, KS = [0 2;1 0];
    case 3, KS = [0 0 3; 0 1 1; 1 0 0];
    case 4, KS = [0 0 0 4;0 0 1 2;0 0 2 0; 0 1 0 1;1 0 0 0];
    case 5, KS = [0 0 0 0 5;
            0 0 0 1 3;
            0 0 0 2 1;
            0 0 1 0 2;
            0 0 1 1 0;
            0 1 0 0 1;
            1 0 0 0 0];
    otherwise, KS = [0 0 0 0 0 6;
            0 0 0 0 1 4;
            0 0 0 0 2 2;
            0 0 0 0 3 0;
            0 0 0 1 0 3;
            0 0 0 1 1 1;
            0 0 0 2 0 0;
            0 0 1 0 0 2;
            0 0 1 0 1 0;
            0 1 0 0 0 1;
            1 0 0 0 0 0];
end,

KS = fliplr(KS);
