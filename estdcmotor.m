function sys = estdcmotor(TT)
    sys = idss( ...
        [0 1; 0 -0.1], ...
        [0; 0.1], ...
        [1 0], ...
        0 ...
    );
    sys.Structure.A.Free = [0 0; 0 -150];
    sys.Structure.B.Free = [0; 150000];
    sys.Structure.C.Free = 0;
    sys.Structure.D.Free = 0;
    sys.Structure.K.Free = 1;

    sys.Ts = 0;

    sys.StateName = ["Position" "Velocity"];
    sys.InputName = 'u1';
    sys.OutputName = 'y1';

    sys.StateUnit = ["m" "m/s"]';
    sys.InputUnit = 'V';
    sys.OutputUnit = 'm';

    sys = ssest(TT, sys);
end