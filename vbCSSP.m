function [H, Pe, Hfilt, C, FE, removed_ch] = vbCSSP(B, S, Nmax, tol_per, remove_badch, correct_amp)
%--- Inputs ---
% B       : Measurement MEG signals (Nm*Nt)
%           Saturated signals should be removed before vbCSSP
% S       : Basis of the noise space (Nm*Nh)
% Nmax    : Num of max EM-iteration (optional, default=1000)
% tol_per : Relative threshold for free energy [%] (optional, default=0.005%)
% remove_badch : If true, channels with negative or outlier gains will be removed (optional, default=false)
% correct_amp  : If true, amplitudes of C and H will be corrected using SSP estimation (optional, default=true)
%
%--- Outputs ---
% H     : Estimated coefficients of the basis (Nh*Nt)
% Pe    : Projection matrix from B to epsilon (residual space of noise = brain signal)
% Hfilt : Converting matrix from B to Hbar (estimated H)
% C     : Estimated (relative) channel gain matrix (Nm*Nm)
% FE    : Free energy calculated when every E-step finished
%
%--- Other variables ---
% Lam   : Measurement noise precision
% Gamma : Posterior precision of h
%
% 2025/01/24 K.Suzuki Outlier gains (threshold=5MAD) will be rejected as well as negative gains
%
% Copyright (C) 2011, ATR All Rights Reserved.
% K.Suzuki 2024

%% Check inputs
% Num of max iteration
if ~exist('Nmax', 'var') || isempty(Nmax)
    Nmax = 1000;
end

% Relative threshold for free energy
if ~exist('tol_per', 'var') || isempty(tol_per)
    tol_per = 0.005; % [%]
end
tol = tol_per / 100;

% If true, ch with bad (negative or outlier) gain will be removed
if ~exist('remove_badch', 'var') || isempty(remove_badch)
    remove_badch = false;
end

% If true, amplitudes of C and H will be corrected by conventional SSP
if ~exist('correct_amp', 'var') || isempty(correct_amp)
    correct_amp = true;
end

%% Initialization
[nt,nm,nh,covmat,icovmat,rho,K,CS,C2,Rbb,lam,Lam] = inner_init(B,S);

% Init free energy
FE = [];

% Init badch index
removed_ch = false(nm, 1);

%% Start iteration
iem = 0;
while iem < Nmax
    % Increment num of iteration
    iem = iem+1;

    %------- Start E-step (p(H|B)) -------
    Gamma = S'*(C2*Lam)*S + eye(nh,nh); % Posterior precision of H
    iGam = Gamma \ eye(size(Gamma));
    Hfilt = iGam * CS' * Lam;
    H = Hfilt * B; % Posterior mean

    % Calculate sufficient statistics for M-step
    HH = H*H';
    Rhh = nt*iGam + HH; % T*iGam + Sum{h(t)*h(t)'}
    Rhb = H * B';       % Sum{h(t)*b(t)'}
    %-------- End E-step --------

    % Store old Lam to calc FE
    % This is because, we must use variables in the same state
    % Here, we use all vars in E-step when we calc FE
    % (it is really hard to use all vars in M-step)
    Lam_old = Lam;

    %------- Start M-step (p(C|B)) -------
    % Calc inverse posterior precision iPhi
    % iPhi = pinv(diag(diag(S*Rhh*S'*Lam)) + pinv(K));
    % Calc inverse posterior precision iPhi using inverse lemma
    % It is equivalent to above one when K is invertible
    idNRN = diag( 1./diag(S*Rhh*S') );
    %iPhi = K - K * inv(idNRN+Lam*K) * Lam*K;
    iPhi = K - K / (idNRN+Lam*K) * Lam*K;

    Cd = iPhi * diag(S*Rhb*Lam); % Cdbar=<Cd>
    C = diag(Cd); % Cbar=<C>
    CS = C * S; % <C>S
    Cd2 = diag(iPhi) + Cd.^2; % <Cd.^2>=diag(iPhi+Cdbar*Cdbar')
    C2 = diag(Cd2); % <C^2>
    %-------- End M-step --------

    %------- Update Lambda -------
    ECdCd = iPhi+Cd*Cd'; % = <CdCd'>

    BB = sum(dot(B,B)); % sum(b'*b)
    BCNH = sum(dot(B, CS*H)); % sum(b'*C*S*h)
    trNCNR = trace(S'*C2*S*Rhh); % sum( tr( S'*C2*S * (iGam+h*h') ) )
    ilam = (BB - 2*BCNH + trNCNR) / (nt*nm);
    lam = 1 ./ ilam;
    Lam = lam.*eye(nm);

    %-------- End Lambda --------

    %------- Update rho and K -------
    rho = trace(icovmat*ECdCd) / 2*nm;
    % rho = lam;
    % rho = trace(pinv(covmat)*ECdCd) / 2*nm;
    K = covmat .* rho;
    % Update pw for monitoring
    pw = 1 / (rho*nt+1);
    %-------- End rho and K --------

    %------ Start Free energy ------
    e1 = nt * (sum(log(diag(Lam_old))) - inner_logdet(Gamma)); % T log(|Lam|/|Gamma|)
    e2 = B.*repmat(diag(Lam_old),[1,nt]);
    e2 = -e2(:)' * B(:); % -sum(b'*Lam*b)

    Hlong1 = repmat(H, [nh,1]);
    Hlong1 = Hlong1(:); % [h(1)_1 h(1)_2 h(1)_3  h(1)_1 h(1)_2 h(1)_3  h(1)_1 h(1)_2 h(1)_3  h(2)_1 h(2)_2 h(2)_3 ...]
    Hlong2 = repmat(H(:), [1,nh])';
    Hlong2 = Hlong2(:); % [h(1)_1 h(1)_1 h(1)_1  h(1)_2 h(1)_2 h(1)_2  h(1)_3 h(1)_3 h(1)_3  h(2)_1 h(2)_1 h(2)_1 ...]
    Glong  = Gamma(:);
    Glong  = repmat(Glong, [nt,1]);% [g11 g21 g31  g12 g22 g32  g13 g23 g33 ...]

    e3 = sum( Hlong1 .* Glong .* Hlong2 ); % sum(hbar'*Gam*hbar)
    e4 = - trace(ECdCd) / rho; % -1/rho tr(iPhi+CdCd')
    e5 = inner_logdet(iPhi); % log|iPhi|
    e6 = -nm*log(rho); % -Mlog(rho)
    e7 = -(nm*nt + nh*nt + nm) * log(2*pi); % -(MT+LT+M) log(2pi)
    e8 = -nh*nt; % -LT

    FE(end+1) = (e1+e2+e3+e4+e5+e6+e7+e8)/2;
    %------- End Free energy -------

    % Relative diff of FE
    if iem>1
        fediff = abs( (FE(end) - FE(end-1)) / min(abs(FE(end-1:end))) );
    end

    if ~rem(iem, 10), disp(['Itr(' num2str(iem) '): FE = ' num2str(FE(iem)) ', diff(FE) = ' num2str(fediff*100) '[%]']); end

    % If bad gains (negative or outlier) exist, restart iteration by removing badch
    ixr = Cd<0 | inner_isoutlier(Cd,'median',5);
    if any(ixr) && remove_badch
        B(ixr,:) = []; % Remove bad ch
        S(ixr,:) = []; % Remove bad ch
        ixorg = find(~removed_ch); % Original index of active ch
        ixro = ixorg(ixr); % Original index of bad ch
        removed_ch(ixro) = true; % Add to bad ch
        warning(['Bad gains were detected for ch: ' num2str(ixro') newline ...
                 'Restart iteration'])
        iem = 0; % Restart iteration
        [nt,nm,nh,covmat,icovmat,rho,K,CS,C2,Rbb,lam,Lam] = inner_init(B,S); % Init parms
        continue
    end

    % Check convergence
    if iem>5 % Minimum iteration
        if fediff < tol
            disp(['Itr(' num2str(iem) '): EM-algorithm converged with diff(FE) = ' num2str(fediff*100) '[%]']);
            break
        end
    end

end

% Check max iteration
if iem==Nmax
    fediff = abs( (FE(end) - FE(end-1)) / min(abs(FE(end-1:end))) );
    disp(['EM-algorithm reached max iteration: ', num2str(iem)]);
    disp(['diff(FE) = ' num2str(fediff)])
end

if any(removed_ch) && remove_badch
    warning(['Total ' num2str(sum(removed_ch)) ' ch were removed'])
end

%% Compute projection matrices using the latest variables
%  Pe = eye(nm) - CS * iGam * CS' * Lam;
Hfilt = iGam * CS' * Lam; 
Pe = eye(nm) - CS * Hfilt;

%% Amplitude correction for estimated C and H
% Because of indeterminancy between C and H for B = CSH
% Correction using OLS estimation (SSP)
if correct_amp
    Hssp = (S'*S)\S' * B;
    % Derive coefficient based on norm ratio
    coef = norm(Hssp) / norm(H);
    % Coefficient correction
    Cd = Cd ./ coef;
    C = diag(Cd); % Cbar=<C>
    Hfilt = Hfilt .* coef;
    H = Hfilt * B;
end

end % End of function

function logdetX = inner_logdet(X)
[L, U, P] = lu(X);
du = diag(U);
c = det(P) * prod(sign(du));
logdetX = log(c) + sum(log(abs(du)));

end % End of function

function [nt,nm,nh,covmat,icovmat,rho,K,CS,C2,Rbb,lam,Lam]=inner_init(B,S)
% Set time length and num of sensor
nt = size(B,2);
nm = size(B,1);

% Total num of harmonics
nh = size(S,2); 

% Define prior covariance of C as identity matrix
% (L2-regularization)
covmat = eye(nm);
icovmat = covmat;

% Init rho and K (prior covariance of C)
pw = 0.5;
rho = (1-pw) / (pw*nt);
K = covmat .* rho;

%% Initialization
% C is initialized using the svd of the data
% [U,S,~] = svd(B,'econ');
% sd = diag(S);
% CN = U(:,1:Nl)*diag(sd(1:Nl));

% This initialization means first Hbar is conventional HF
% But diagonal loading const is added
% Assuming C=eye
CS = S;
C2 = eye(nm,nm);

% The noise precision is initialized using the diagonal elements
% of the data covariance
Rbb = B*B'; % Sum{b(t)*b(t)'}
lam = 1./diag(Rbb/nt);
% Lam = diag(lam);
Lam = mean(lam).*eye(nm); % Use scalar lam

% lam = 0.1; % Simple case
% Lam = lam*eye(nm);
end % End of function

function outlier = inner_isoutlier(x, method, p)

if ~exist('method','var') || isempty(method)
    method = 'median';
end

if ~exist('p','var') || isempty(p)
    p = 3;
end

switch method
    case 'median'
        % thrsh = p*MAD
        c = -1 /(sqrt(2)*erfcinv(3/2)); % ~1.4826
        xcenter = median(x,1,'omitnan');
        xmad = c*median(abs(x - xcenter), 1, 'omitnan');

        lowerbound = xcenter - p*xmad;
        upperbound = xcenter + p*xmad;
    
    case 'mean'
        % thresh = p*STD
        xcenter = mean(x,1,'omitnan');
        xstd = std(x,0, 1,'omitnan');

        lowerbound = xcenter - p*xstd;
        upperbound = xcenter + p*xstd;
end

outlier = x>upperbound | x<lowerbound;

end % End of function

