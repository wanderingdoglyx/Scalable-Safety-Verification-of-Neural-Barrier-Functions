function UNSAT = nonlinear_prog_checking(W_overl, r_overl, ...
    B_act_w, B_act_r, ...
    B_inact_w, B_inact_r)
    W_Bound = -B_act_w + B_inact_w;
    r_Bound = -B_act_r - B_inact_r;
    W_overl = -W_overl;
    r_overl = -r_overl;
    x0 = [0;0];
    options = optimoptions('fmincon','Display','iter','Algorithm','sqp');
    obj = @(x)W_overl(1)*(x(2)+2*x(1)*x(2))+W_overl(2)*(-x(1)+2*x(1)^2-x(2)^2)
%     obj = @(x)(1/3)*W_overl(2)*x(1)^3-W_overl(2)*x(1)+(W_overl(1)-W_overl(2))*x(2);
    % dbdxf=(1/3)*W_overl(2)*x1^3-W_overl(2)*x1+(W_overl(1)-W_overl(2))*x2;
    nonlcon = @circlecon;
    [x,fval] = fmincon(obj,x0,W_Bound,-r_Bound,-W_overl,r_overl,[-2,-2],[2,2])
    if fval>=0
        UNSAT = false;
    else
        UNSAT = true;
    end