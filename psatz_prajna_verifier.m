function UNSAT = psatz_prajna_verifier(W_overl, r_overl, ...
    B_act_w, B_act_r, ...
    B_inact_w, B_inact_r)
    pvar x1 x2;

    W_overl = -W_overl;
    r_overl = -r_overl;
    W_Bound = B_act_w + B_inact_w;
    r_Bound = -B_act_r - B_inact_r;
    
    x = [x1 ; x2];
    
    prog = sosprogram(x);
    degree = 6;
    [prog,eta] = sospolyvar(prog,monomials(x,1:degree),'wscoeff');
    [prog,theta] = sospolyvar(prog,monomials(x,1:degree),'wscoeff');
    [prog,alpha0] = sossosvar(prog,monomials(x,1:degree),'wscoeff');
    [prog,alpha] = sossosvar(prog,monomials(x,1:degree));
    [prog,alpha1] = sossosvar(prog,monomials(x,1:degree));
    [prog,alpha2] = sossosvar(prog,monomials(x,1:degree));
    [prog,omega] = sospolyvar(prog,monomials(x,1:degree),'wscoeff');
    [prog,beta] = sossosvar(prog,monomials(x,1:degree));
    [prog,r1] = sospolyvar(prog,1);
    [prog,r2] = sospolyvar(prog,1);
    
%     b = W_overl*x + r_overl;
    b = W_overl*x + r_overl;
%     Bound = W_Bound*x + r_Bound;
    Bound = W_Bound*x + r_Bound;
    
    dbdxf=(1/3)*W_overl(2)*x1^3-W_overl(2)*x1+(W_overl(1)-W_overl(2))*x2;
    dbdxg=0;
    % prog =soseq(prog, eta*b + alpha0 - alpha*dbdxf  ...
    % + alpha1*p + dbdxf^2 + p.^2);
    prog =soseq(prog, eta*b + beta*Bound(2) + alpha0 - alpha1*dbdxf + ...
        alpha2*Bound + alpha*dbdxf*Bound + dbdxf^2);
    % prog = sossetobj(prog,1);
    
    solver_opt.solver = 'sdpt3';
    [prog,info] = sossolve(prog,solver_opt);
    UNSAT = info.pinf;

end