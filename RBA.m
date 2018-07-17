classdef RBA < handle
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
    % Model free mapping tool.
    %
    % rba = RBA(params) creates an instance of the RBA class.
    % params is a structure with 7 required fields
    %   - f_sampling: sampling frequency (1/TR)
    %   - n_samples : number of samples (volumes)
    %   - n_rows    : number of rows (in-plane resolution)
    %   - n_cols    : number of columns (in-plance resolution)
    %   - n_slices  : number of slices
    %
    % this class has the following function
    %
    %   - delay = RBA.get_delay();
    %   - direction = RBA.get_direction(direction);
    %   - RBA.set_delay(delay);
    %   - RBA.set_direction(direction);
    %   - results = RBA.fitting(data);
    %
    % use help RBA.function to get more detailed help on any specific
    % function (e.g. help RBA.fitting)
    %
    % typical workflow:
    % 1. rba = RBA(params);
    % 2. rba.set_delay(delay);
    % 3. rba.set_direction(direction);
    % 4. results = rba.perform_ridge(data,lambda);
    
    properties (Access = private)
        
        is
        % functions
        two_gamma               % two gamma hrf function
        
        % parameters
        f_sampling
        p_sampling
        n_samples
        n_predictors
        n_rows
        n_cols
        n_slices
        n_total
        l_hrf
        hrf
        X
        lambda
        
    end
    
    methods (Access = public)
        
        function self = RBA(params,varargin)
            % constructor
            addpath(pwd)
            p = inputParser;
            addRequired(p,'params',@isstruct);
            addOptional(p,'hrf',[]);
            p.parse(params,varargin{:});
            
            self.is = 'RBA tool';
            
            
            self.two_gamma = @(t) (6*t.^5.*exp(-t))./gamma(6)...
                -1/6*(16*t.^15.*exp(-t))/gamma(16);
            
            self.f_sampling = params.f_sampling;
            self.p_sampling = 1/self.f_sampling;
            self.n_samples = params.n_samples;
            self.n_predictors = 0;
            self.n_rows = params.n_rows;
            self.n_cols = params.n_cols;
            self.n_slices = params.n_slices;
            self.n_total = self.n_rows*self.n_cols*self.n_slices;
            self.X = [];
            
            if ~isempty(p.Results.hrf)
                self.l_hrf = size(p.Results.hrf,1);
                if ndims(p.Results.hrf)==4
                    self.hrf = reshape(p.Results.hrf,...
                        self.l_hrf,self.n_total);
                else
                    self.hrf = p.Results.hrf;
                end
                
            else
                self.hrf = self.two_gamma(0:self.p_sampling:34)';
                self.l_hrf = numel(self.hrf);
            end
        end
        
        
        function hrf = get_hrf(self)
            % returns the hemodynamic response used by the class.
            % If a single hrf is used for every voxel, this function returns a column vector.
            % If a unique hrf is used for each voxel, this function returns a 4-dimensional matrix with columns corresponding to time.
            if size(self.hrf,2)>1
                hrf = reshape(self.hrf,self.l_hrf,...
                    self.n_rows,self.n_cols,self.n_slices);
            else
                hrf = self.hrf;
            end
        end
        
        function design = get_design(self)
            % returns the design matrix.
            design = self.X;
        end
        
        
        function set_hrf(self,hrf)
            % replace the hemodynamic response with a new hrf column vector.
            self.l_hrf = size(hrf,1);
            self.hrf = hrf;
        end
        
        
        function set_design(self,X,varargin)
            % provide a t-by-p design matrix to the class with t timepoints
            % and p predictors.
            
            p = inputParser;
            addRequired(p,'X',@isnumeric);
            addOptional(p,'convolve',false);
            p.parse(X,varargin{:});
            
            c = p.Results.convolve;
            self.n_predictors = size(p.Results.X,2);
            if c
                X_fft = fft(p.Results.X);
                hrf_fft = fft(repmat([self.hrf;...
                    zeros(self.n_samples-self.l_hrf,1)],...
                    [1,self.n_predictors]));
                self.X = zscore(ifft(X_fft.*hrf_fft));
            else
                self.X = zscore(p.Results.X);
            end
        end
        
        function optimize_lambda(self,data)
            
            text = 'optimizing lambda...';
            fprintf('%s\n',text)
            wb = waitbar(0,text,'Name',self.is);
            
            data = zscore(reshape(data(1:self.n_samples,:,:,:),...
                self.n_samples,self.n_total));
            K = 3:9;
            if isprime(self.n_samples)
                data(end,:) = [];
                id = mod(self.n_samples-1,K)==0;
                K = min(K(id));
                if isempty(K)
                    K = 2;
                end
                samples = (self.n_samples-1) / K;
            else
                id = mod(self.n_samples,K)==0;
                K = min(K(id));
                if isempty(K)
                    K = 2;
                end
                samples = self.n_samples / K;
            end
            fprintf(' using %i splits\n',K)
            M = 0:5;
            fit = zeros(6,self.n_total);
            for u=1:6
                for k=0:K-1
                    s = k * samples+1 : k * samples + samples;
                    tst_X = self.X;
                    tst_X(s,:) = [];
                    tst_data = data;
                    tst_data(s,:) = [];
                    trn_X(s,:) = self.X(s,:);
                    trn_data = data(s,:);
                    
                    
                    mag_d = sqrt(sum(tst_data.^2));
                    
                    if self.n_samples<self.n_predictors
                        [U,D,V] = svd(trn_X,'econ');
                        XX = V * inv(D^2 + ...
                            10^M(u) * eye(samples)) * D * U';
                    else
                        XX = (trn_X'* trn_X + ...
                            10^M(u) * eye(self.n_predictors)) \ trn_X';
                    end
                    
                    for v=1:self.n_total
                        b = XX * trn_data(:,v);
                        y = tst_X * b;
                        mag_y = sqrt(y'* y);
                        fit(u,v) = fit(u,v) + ((y'* tst_data(:,v))...
                            / (mag_y * mag_d(v))) / (K+1);
                    end 
                end
                waitbar(u/6,wb)
            end
            [~,id] = max(mean(fit,2));
            self.lambda = 10^M(id);
        end
        
        function results = perform_ridge(self,data,varargin)
            % performs ridge regression and returns a structure with the following fields
            %  - Beta
            %  - RSS
            %  - F_statistic
            %  - P_value
            %
            % the dimension of each field corresponds to the dimension of the data.
            %
            % required input is
            %  - data  : a 4-dimensional matrix of empirically observed BOLD timecourses.
            %               Columns correspond to time (volumes).
            %
            % optional inputs are
            %  - lambda: shrinkage parameter
            %  - mask  : mask file for selecting voxels
            
            text = 'performing ridge regression...';
            fprintf('%s\n',text)
            wb = waitbar(0,text,'Name',self.is);
            
            p = inputParser;
            addRequired(p,'data',@isnumeric);
            addOptional(p,'lambda',[]);
            addOptional(p,'mask',true(self.n_total,1));
            p.parse(data,varargin{:});
            
            data = p.Results.data;
            if ~isempty(p.Results.lambda)
                self.lambda = p.Results.lambda;
            end
            mask = reshape(p.Results.mask,self.n_total,1);
            
            if self.n_samples<self.n_predictors
                [U,D,V] = svd(self.X,'econ');
                XX = V * inv(D^2 + ...
                    self.lambda * eye(self.n_samples)) * D * U';
            else
                XX = (self.X'* self.X + ...
                    self.lambda * eye(self.n_predictors)) \ self.X';
            end
            data = zscore(reshape(data(1:self.n_samples,:,:,:),...
                self.n_samples,self.n_total));
            
            results.Beta = cell(self.n_rows,self.n_cols,self.n_slices);
            results.RSS = zeros(self.n_total,1);
            results.F_stat = zeros(self.n_total,1);
            results.P_value = ones(self.n_total,1);
            
            df1 = 2;
            df2 = self.n_samples-1;
            for v=1:self.n_total
                if mask(v)
                b = XX * data(:,v);
                y = self.X * b;
                y_ = mean(y);
                MSM = (y-y_)'*(y-y_)/df1;
                MSE = (y-data(:,v))'*(y-data(:,v))/df2;
                
                results.Beta{v} = b;
                results.RSS(v) = (y-data(:,v))'*(y-data(:,v));
                results.F_stat(v) = MSM/MSE;
                results.P_value(v) = 1-fcdf(MSM/MSE,df1,df2);
                end
                waitbar(v/self.n_total,wb)
            end
            
            results.RSS = reshape(results.RSS,...
                self.n_rows,self.n_cols,self.n_slices);
            results.F_stat = reshape(results.F_stat,...
                self.n_rows,self.n_cols,self.n_slices);
            results.P_value = reshape(results.P_value,...
                self.n_rows,self.n_cols,self.n_slices);
            
            close(wb)
            fprintf('done\n');
        end
        
    end
    
end