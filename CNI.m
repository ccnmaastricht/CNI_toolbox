classdef CNI < handle
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % %%%                               LICENSE                             %%%
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    % Copyright 2018 Mario Senden
    %
    % This program is free software: you can redistribute it and/or modify
    % it under the terms of the GNU Lesser General Public License as published
    % by the Free Software Foundation, either version 3 of the License, or
    % (at your option) any later version.
    %
    % This program is distributed in the hope that it will be useful,
    % but WITHOUT ANY WARRANTY; without even the implied warranty of
    % MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    % GNU Lesser General Public License for more details.
    %
    % You should have received a copy of the GNU Lesser General Public License
    % along with this program.  If not, see <http://www.gnu.org/licenses/>.
    %
    %
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % %%%                             DESCRIPTION                           %%%
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    % rm = RM() creates an instance of the recurrent model of perceptual
    % learning in early visual cortex using standard parameter values
    % see: Lange G, Senden M, Radermacher A, De Weerd P.
    % Interference  with highly trained skills reveals  competition
    % rather than consolidation (submitted).
    %
    % Use rm.set_OD(x) to set orientation difference to value 'x'; if no value
    %     is provided, OD will be reset to its baseline state (7.5 unless
    %     specified otherwise during construction)
    % Use rm.set_PHI(x) to set reference orientation to value 'x'; if no value
    %     is provided, PHI wil be reset to ts baseline state (135° unless
    %     specified otherwise during construction)
    % Use rm.fix(P) to fix a proportion 'P' of connection weights.
    % Use rm.get_JND() to read out the current JND
    % Use rm.session() to simulate a single session of staircase experiment.
    % Use rm.reset() to restore the model to its naive state.
    
    properties (Access = private)
        % functions
        two_gamma               % two gamma hrf function
        
        % parameters
        f_sampling
        p_sampling
        f_stimulation
        n_voxels
        n_samples
        l_signal
        l_hrf
        hrf
        
    end
    
    methods (Access = public)
        % constructor
        function self = CNI(params,varargin)
            p = inputParser;
            addRequired(p,'params',@isstruct);
            addOptional(p,'hrf',[]);
            p.parse(params,varargin{:});
            
            self.two_gamma = @(t) (6*t.^5.*exp(-t))./gamma(6)...
                -1/6*(16*t.^15.*exp(-t))/gamma(16);
            
            self.f_sampling = p.Results.params.f_sampling;
            self.p_sampling = 1/self.f_sampling;
            self.f_stimulation = p.Results.params.f_stimulation;
            self.n_voxels = p.Results.params.n_voxels;
            self.n_samples = p.Results.params.n_samples;
            self.l_signal = self.n_samples*self.p_sampling;
            if ~isempty(p.Results.hrf)
                self.hrf = p.Results.hrf;
            else
                self.hrf = self.two_gamma(0:self.p_sampling:34)';
            end
            self.l_hrf = numel(self.hrf);
        end
        
        % phase-lag analysis
        function results = phase_lag(self,data)
            t = (0:self.p_sampling:self.l_signal-1)';
            X = [cos(2*pi*t*self.f_stimulation),...
                sin(2*pi*t*self.f_stimulation)];
            X_fft = fft(X);
            hrf_fft = fft(repmat([self.hrf;...
                zeros(self.n_samples-self.l_hrf,1)],[1,2]));
            X = ifft(X_fft.*hrf_fft);
            XX = (X'*X)\X';
            
            results.phase = zeros(self.n_voxels,1);
            results.p = zeros(self.n_voxels,1);
            
            df1 = 2;
            df2 = self.n_samples-1;
            for v=1:self.n_voxels
                b = XX*data(:,v);
                y = X*b;
                y_ = mean(y);
                MSM = (y-y_)'*(y-y_)/df1;
                MSE = (y-data(:,v))'*(y-data(:,v))/df2;
                F = MSM/MSE;
                results.phase(v) = atan2(b(2),b(1));
                results.p(v) = 1-fcdf(F,df1,df2);
            end 
        end
        
        % tuning model analysis
        function results = tuning_model(self,data,PARAMS,fun)
            results = fun(2); 
        end
    end
    
end


