a = dir('d50_*');
x=[];
comps = 32;
for i=1:length(a)
m = importdata(a(i).name);
m.data = [m.data,zeros(size(m.data,1),(length(m.textdata(1:comps+2))-size(m.data,2)))];
x=[x;m.data(2:end,1:comps+2)];
end