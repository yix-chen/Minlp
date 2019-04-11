% matlab
% minlpsolve_sca : function description
function [x, p, maxt, exitflag] = minlpsolve_sca (num_user, num_bs, gain, power_bs, bandw, x_d)



alpha = ones(num_user, num_bs);
beta = zeros(num_user, num_bs);
save alphaandbeta.mat


[var_awgn] = const(num_user, num_bs, gain, power_bs, bandw, x_d);

% check feasibility
x_d_k = ceil(length(x_d) / num_bs);
x_d_l = (length(x_d) - (x_d_k - 1) * num_bs) * (x_d_k ~= 0);
delta = 1e-3;

flag = (x_d_l ~= num_bs) * (x_d_k ~= 0); % flag = 1 , hang wei man
isfeasible = 1;
for k = 1 : x_d_k - flag
	sum = 0;
	for l = 1 : num_bs
		sum = sum + x_d((k - 1) * num_bs + l);
	end
	if abs(sum-1) > delta
		isfeasible = 0;
		break;
	end
end

if flag == 1
	k = x_d_k;
	sum = 0;
	for l = 1 : x_d_l
		sum = sum + x_d((k - 1) * num_bs + l);
    end
    
	if sum > 1
		isfeasible = 0;
	end
end

if isfeasible == 0
	x = [];
	p = [];
	maxt = -1e-6;
	exitflag = -2;
	return;
end

% variable: [x_ij(wave), p_j�??? t(non-linear)]
n_d = length(x_d);
var_number = num_user * num_bs + num_bs + 1 - n_d;

% C1 & C3 - non-linear�???该�????@min_t�???
nonlcon = @min_t;

% C2
Aeq = zeros(num_user - x_d_k + flag, var_number);
beq = zeros(num_user - x_d_k + flag, 1);

if flag == 1
	for i = 1 : num_bs - x_d_l
		Aeq(1, i) = 1;
	end
	beq(1) = 1 - sum;
	for k = 1 : num_user - x_d_k
		for l = 1 : num_bs
			Aeq(k + 1, num_bs - x_d_l + (k-1) * num_bs + l) = 1;
		end
		beq(k + 1) = 1;
	end
else
	for k = 1 : num_user - x_d_k
		for l = 1 : num_bs
			Aeq(k, (k - 1) * num_bs + l) = 1;
		end
		beq(k) = 1;
	end
end


% C4
lb = zeros(1, var_number);
ub = ones(1, var_number);
lb(num_user * num_bs + 1 - n_d : end) = -Inf;
ub(num_user * num_bs + 1 - n_d : end-1) = log(power_bs);
ub(end) = Inf;

% Initial-x0
x0 = zeros(1,var_number);

% function
fun = @(x)-x(end);
A = [];
b = [];

% options = optimoptions('fmincon','MaxIterations',1e5,'MaxFunctionEvaluations',1e4,'TolFun',1e-4,'Display','iter','Algorithm','sqp');
options = optimoptions('fmincon','MaxIterations',1e5,'MaxFunctionEvaluations',2e5,'TolFun',1e-4);

tolerance = 1e-3;
Tmax = 1e3; % max iteration number
convergence = false;
iter_count = 0;

while convergence == false && iter_count < Tmax

	[x, fval, exitflag, ~] = fmincon(fun, x0, A, b, Aeq, beq, lb, ub, nonlcon, options);
	xwave = x(1 : num_user*num_bs - n_d);
	pwave = x(num_user * num_bs + 1 - n_d : num_user * num_bs + num_bs - n_d);
	p = exp(pwave);
	gma_numerical = zeros(num_user, num_bs);
	for i = 1 : num_user
		for j = 1 : num_bs
			gma_numerical(i, j) = p(j) * gain(i, j) / (var_awgn.^2 + get_interf_numerical(p, i, j, gain));
		end
	end

	prior_alpha = alpha;
	prior_beta = beta;
	alpha = gma_numerical ./ (1 + gma_numerical);
	beta = log2(1 + gma_numerical) - (gma_numerical ./ (1 + gma_numerical)) .* log2(gma_numerical);

	save('alphaandbeta.mat', 'alpha', 'beta')
	nonlcon = @min_t;
	iter_count = iter_count + 1;
	if any(any(abs(alpha - prior_alpha) > tolerance)) || any(any(abs(beta - prior_beta) > tolerance))
		convergence = false;
	else
		convergence = true;
	end

end


maxt = -fval;
x = [x_d,xwave];
%eqs_for_C3andC4 = num_user * num_bs + num_bs;
%A = zeros(eqs_for_C3andC4 * 2, var_number);
%B = zeros(eqs_for_C3andC4 * 2);

%for i = 1 : eqs_for_C3andC4
%	A(i, i) = 1;
%end

%for i = eqs_for_C3andC4 + 1 : eqs_for_C3andC4 * 2
%	A(i, i - eqs_for_C3andC4) = -1�??? 

%B(1 : num_user*num_bs) = 1;
%B(num_user*num_bs+1 : num_user*num_bs+num_bs) = _power;
%B(eqs_for_C3andC4+1 : eqs_for_C3andC4+num_user*num_bs) = 0;
%B(eqs_for_C3andC4+num_user*num_bs+1 : end) = 0;






