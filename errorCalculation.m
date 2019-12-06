clc 
close all
clear all

a = load('gpu_d50_data.mat');
b = load('correct_mpi1Data.mat');
c = load('correct_mpi2Data.mat');
d = load('correct_mpi4Data.mat');

gdata = d.m8_4;
mdata = b.m8_1;
comps = 8;

gdata(any(isinf(gdata),2),:) = [];

mpilen = length(mdata(:,1));
gpulen = length(gdata(:,1));

maxln = max(mpilen,gpulen);
minln = min(mpilen,gpulen);
x = 1;
toterr = 0;

tolerr = 1e-4;
for i = 1:maxln
    for j = x:minln
        if (mpilen > gpulen)
            if (abs(mdata(i,2) - gdata(j,2)) < tolerr)
                for k = 1:comps
                    toterr = toterr + (mdata(i,comps+2) - gdata(j,comps+2))^2 / gpulen;
                    break
                end
            end
        else 
            if (abs(mdata(j,2) - gdata(i,2)) < tolerr)
                for k = 1:comps
                    toterr = toterr + (mdata(j,comps+2) - gdata(i,comps+2))^2 / gpulen;
                    break
                end
            end
        end
    end
end

toterr