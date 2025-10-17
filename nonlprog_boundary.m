function UNSAT = nonlprog_boundary(W_overl0, r_overl0, ...
    W_overl1, r_overl1, ...
    W_Bound0, r_Bound0)
    x0 = [0; -1.8; 0];
    v=1;
    options = optimoptions('fmincon','Display','iter','Algorithm','sqp');

    W_overl = -W_overl0;
    r_overl = -r_overl0;
%     u = @(x)-sin(x(3))+3*((x(1)*sin(x(3))+x(2)*cos(x(3)))/(0.5+x(1)^2+x(2)^2));
    obj = @(x)W_overl(1)*v*sin(x(3)) ...
        + W_overl(2)*v*cos(x(3)) ...
        + W_overl(3)*(-sin(x(3))+3*((x(1)*sin(x(3))+x(2)*cos(x(3)))/(0.5+x(1)^2+x(2)^2)));
%     obj = @(x)W_overl(1)*(x(2)+2*x(1)*x(2))+W_overl(2)*(-x(1)+2*x(1)^2-x(2)^2)
%     obj = @(x)(1/3)*W_overl(2)*x(1)^3-W_overl(2)*x(1)+(W_overl(1)-W_overl(2))*x(2);
    % dbdxf=(1/3)*W_overl(2)*x1^3-W_overl(2)*x1+(W_overl(1)-W_overl(2))*x2;
    nonlcon = @circlecon;
    [x,fval] = fmincon(obj,x0,[],[],[W_Bound0;-W_overl],[-r_Bound0;r_overl],[-2,-2,-1.57],[2,2,1.57])
    if fval>=0
        UNSAT = false;
    else
        W_overl = -W_overl1;
        r_overl = -r_overl1;
        obj = @(x)W_overl(1)*v*sin(x(3)) ...
        + W_overl(2)*v*cos(x(3)) ...
        + W_overl(3)*(-sin(x(3))+3*((x(1)*sin(x(3))+x(2)*cos(x(3)))/(0.5+x(1)^2+x(2)^2)));
%         obj = @(x)W_overl(1)*(x(2)+2*x(1)*x(2))+W_overl(2)*(-x(1)+2*x(1)^2-x(2)^2)
%         obj = @(x)(1/3)*W_overl(2)*x(1)^3-W_overl(2)*x(1)+(W_overl(1)-W_overl(2))*x(2);
        % dbdxf=(1/3)*W_overl(2)*x1^3-W_overl(2)*x1+(W_overl(1)-W_overl(2))*x2;
        nonlcon = @circlecon;
        [x,fval] = fmincon(obj,x0,[],[],[W_Bound0;-W_overl],[-r_Bound0;r_overl],[-2,-2,-1.57],[2,2,1.57])
        if fval>=0
            UNSAT = false;
        else
            UNSAT = true;
        end
    end
end