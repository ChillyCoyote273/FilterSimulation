function TT = loadtimetable(filepath)
    table = readtable(filepath);
    ts = table.t;
    times = linspace(ts(1), ts(length(ts)), length(ts));
    for i = 1 : width(table)
        table{:,i} = interp1(ts, table.(i), times).';
    end
    TT = table2timetable(table, 'TimeStep', seconds(times(2) - times(1)));
end