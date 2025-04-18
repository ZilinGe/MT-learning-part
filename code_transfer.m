%This Matlab script can be used to reproduce Figures 7.2(a) and 7.2(b) in the monograph:
%
%Ozlem Tugfe Demir, Emil Bjornson and Luca Sanguinetti (2021),
%"Foundations of User-Centric Cell-Free Massive MIMO", 
%Foundations and Trends in Signal Processing: Vol. 14: No. 3-4,
%pp 162-472. DOI: 10.1561/2000000109
%
%This is version 1.0 (Last edited: 2021-01-31)
%
%License: This code is licensed under the GPLv2 license. If you in any way
%use this code for research that results in publications, please cite our
%monograph as described above.

%Empty workspace and close figures
close all;
clear;


%% Define simulation setup

%Number of setups with random UE locations
nbrOfSetups = 1400;

%Number of channel realizations per setup
nbrOfRealizations = 1000;

%Number of APs in the cell-free network
L = 100;

%Number of antennas per AP
N = 4;

%Number of UEs in the network
K = 40;

%Length of the coherence block
tau_c = 200;

%Compute number of pilots per coherence block
tau_p = 10;

%Compute the prelog factor assuming only downlink data transmission
preLogFactor = (tau_c-tau_p)/tau_c;

%Angular standard deviation in the local scattering model (in radians)
ASD_varphi = deg2rad(15);  %azimuth angle
ASD_theta = deg2rad(15);   %elevation angle

%Total uplink transmit power per UE (mW)
p = 100;

%Total downlink transmit power per AP (mW)
rho_tot = 200;


%Prepare to save simulation results for centralized downlink operation with
%P-MMSE precoding
SE_DL_PMMSE_fractional2b = zeros(K,nbrOfSetups); %FPA, \upsilon = -0.5, \kappa = 0.5

Ptot_record = zeros(nbrOfSetups, 1);

%% Go through all setups
for n = 1:nbrOfSetups
    
    %Display simulation progress
    disp(['Setup ' num2str(n) ' out of ' num2str(nbrOfSetups)]);
    
    %Generate one setup with UEs at random locations
    [gainOverNoisedB,R,pilotIndex,D,D_small] = generateSetup(L,K,N,tau_p,1,0,ASD_varphi,ASD_theta);
    
    
    %Generate channel realizations, channel estimates, and estimation
    %error correlation matrices for all UEs to the cell-free APs
    [Hhat,H,B,C] = functionChannelEstimates(R,nbrOfRealizations,L,K,N,tau_p,pilotIndex,p);
    
    
    % Full uplink power for the computation of precoding vectors using
    % virtual uplink-downlink duality
    p_full = p*ones(K,1);
   
    
    %Obtain the expectations for the computation of the terms in
    %(7.13)-(7.15) 
    [signal_P_MMSE, signal2_P_MMSE, scaling_P_MMSE,...
        signal_P_RZF, signal2_P_RZF, scaling_P_RZF,...
        signal_LP_MMSE,signal2_LP_MMSE, scaling_LP_MMSE] = ...
        functionComputeExpectations(Hhat,H,D,C,nbrOfRealizations,N,K,L,p_full);
    
    %Compute the terms in (7.13)-(7.15)
    bk_PMMSE = zeros(K,1);
    ck_PMMSE = signal2_P_MMSE.';

    for k = 1:K
        
        bk_PMMSE(k) = abs(signal_P_MMSE(k,k))^2;
      
        ck_PMMSE(k,k) = ck_PMMSE(k,k) - bk_PMMSE(k);

    end
    
    %Obtain the scaling factors for the precoding vectors and scale the
    %terms in (7.13)-(7.15) accordingly
    sigma2_PMMSE = sum(scaling_P_MMSE,1).';
   
    for k = 1:K
        
        bk_PMMSE(k) = bk_PMMSE(k)/sigma2_PMMSE(k);
        
        ck_PMMSE(k,:) = ck_PMMSE(k,:)/sigma2_PMMSE(k);
    
    end
    
    %Scale the expected values of the norm squares of the portions of the centralized precoding in
    %accordance with the normalized precoding vectors in (7.16)
    portionScaling_P_MMSE = scaling_P_MMSE./repmat(sum(scaling_P_MMSE,1),[L 1]);

    %Compute the fractional power allocation for centralized precoding
    %according to (7.43) with different \upsilon and \kappa parameters
    rho_fractional = functionCentralizedPowerAllocation(K,gainOverNoisedB,D,rho_tot,portionScaling_P_MMSE,-0.5,0.5);


    % cal SINR
    SINR = bk_PMMSE .* rho_2c ./ (ck_PMMSE' * rho_2c + 1);
    SINR_th_db = 10 * log10(mean(SINR));

    % cal c cofficients
    c = compute_c_coefficients(tau_p, tau_c, fs, Ts, ...
        NDFT, Nused ,BW, SINR_th_db);

    %cal totoal power
    Ptot_record(n) = computePtot(rho_2c, N_ap, c,D);


    %Compute SEs according to Theorem 6.1 with several scalable power
    %allocation schemes
    SE_DL_PMMSE_fractional2b(:,n) = preLogFactor*log2(1+bk_PMMSE.*rho_fractional./(ck_PMMSE'*rho_fractional+1));



    
end
