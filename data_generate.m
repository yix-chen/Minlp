% data generate
clear,clc
addpath(genpath(pwd));

num_user = 4;
num_bs = 2;
num_instance = 10;


bandw = 10^6; % 1MHz



R1 = 250; R2 = 500; % user
r1 = 15;  r2 = 50; % Bs

Gain_UB = zeros(num_user, num_bs, num_instance);
power_Vector = zeros(num_instance, num_bs);

i = 1;
while i <= num_instance
	power_bs = rand(1, num_bs) * 10 + 30; % 30 ~ 40 w
	[x_bs, y_bs] = create_random_location(r2, r1, num_bs, 0, 0);
	[x_user, y_user] = create_random_location(R2, R1, num_user, 0, 0);	
	gain = Fun_chGain_MultiBS(num_user, num_bs, x_user, y_user, x_bs, y_bs);
	x_d = [];
    
	[x, p, maxt, exitflag] = minlpsolve_sca(num_user, num_bs, gain, power_bs, bandw, x_d);
    
	if exitflag == 1 || exitflag == 2
		Gain_UB(:, :, i) = gain;
		power_Vector(i, :) = power_bs;
        i = i + 1;
	end

end

if ~exist('train', 'dir')
	mkdir('train')
end

clear power_bs x_bs y_bs x_user y_user x
clear R1 R2 r1 r2 gain x_d p maxt exitflag

save('./train/train.mat')
	