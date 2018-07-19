classdef RRT < handle
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
    % Ridge regression tool.
    %
    % rrt = RRT(params) creates an instance of the RRT class.
    % params is a structure with 5 required fields
    %   - f_sampling: sampling frequency (1/TR)
    %   - n_samples : number of samples (volumes)
    %   - n_rows    : number of rows (in-plane resolution)
    %   - n_cols    : number of columns (in-plance resolution)
    %   - n_slices  : number of slices
    %
    % optional inputs are
    %   - hrf       : either a column vector containing a single hemodynamic
    %                 response used for every voxel;
    %                 or a matrix with a unique hemodynamic response along
    %                 its columns for each voxel.
    %                 By default the canonical two-gamma hemodynamic response
    %                 function is generated internally based on the scan parameters.
    %
    % this class has the following functions
    %
    %   - hrf = RRT.get_hrf();
    %   - X = RRT.get_design();
    %   - RRT.set_hrf(hrf);
    %   - RRT.set_design(X);
    %   - RRT.optimize_lambda(data,range);
    %   - results = RRT.perform_ridge(data);
    %
    % use help RRT.function to get more detailed help on any specific
    % function (e.g. help RRT.perform_ridge)
    %
    % typical workflow:
    % 1. rrt = RRT(params);
    % 2. rrt.set_design(X);
    % 3. rrt.optimize_lambda(data,range);
    % 4. results = rrt.perform_ridge(data);
    
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
        
        function self = RRT(params,varargin)
            % constructor
            p = inputParser;
            addRequired(p,'params',@isstruct);
            addOptional(p,'hrf',[]);
            p.parse(params,varargin{:});
            
            self.is = 'RRT tool';

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
            hrf = self.hrf;
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
            % provide a n-by-p design matrix to the class with n samples
            % and p predictors.
            %
            % optional inputs is
            %  - convolve: a logical (boolean) value indicating whether the
            %              design matrix needs to be convolved with the
            %              hemodynamic response function (default = false)
            
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
        
        function optimize_lambda(self,data,range,varargin)
            % performs k-fold cross-validation to find an optimal value
            % for the shrinkage parameter lambda.
            %
            % required inputs are
            %  - data : a matrix of empirically observed BOLD timecourses
            %            whose columns correspond to time (volumes).
            %  - range: a range of candidate values for lambda.
            %
            % optional inputs is
            %  - mask  : mask file for selecting voxels
            
            text = 'optimizing lambda...';
            fprintf('%s\n',text)
            wb = waitbar(0,text,'Name',self.is);
            
            p = inputParser;
            addRequired(p,'data',@isnumeric);
            addRequired(p,'range',@isnumeric);
            addOptional(p,'mask',true(self.n_total,1));
            p.parse(data,range,varargin{:});
            
            range = p.Results.range;
            mask = reshape(p.Results.mask,self.n_total,1);
            
            data = zscore(reshape(p.Results.data(1:self.n_samples,:,:,:),...
                self.n_samples,self.n_total));
            data = data(:,mask);
            numV = size(data,2);
            K = 2:7;
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
            fprintf('->using %i splits\n',K)
            iterations = numel(range);
            fit = zeros(iterations,numV);
            total = iterations*K*numV;
            for i=1:iterations
                fprintf('-->lambda = %.2f\n',range(i))
                
                for k=0:K-1
                    fprintf('--->split %i\n',k+1)
                    s = k * samples+1 : k * samples + samples;
                    trn_X = self.X;
                    trn_X(s,:) = [];
                    trn_data = data;
                    trn_data(s,:) = [];
                    tst_X = self.X(s,:);
                    tst_data = data(s,:);
                    
                    mag_d = sqrt(sum(tst_data.^2));
                    
                    if self.n_samples<self.n_predictors
                        [U,D,V] = svd(trn_X,'econ');
                        XX = V * inv(D^2 + ...
                            range(i) * eye(samples * (K-1))) * D * U';
                    else
                        XX = (trn_X'* trn_X + ...
                            range(i) * eye(self.n_predictors)) \ trn_X';
                    end
                    
                    for v=1:numV
                        b = XX * trn_data(:,v);
                        y = tst_X * b;
                        mag_y = sqrt(y' * y);
                        fit(i,v) = fit(i,v) + ((y'* tst_data(:,v))...
                            / (mag_y * mag_d(v)) - fit(i,v)) / (K+1);
                        waitbar(((i-1) * numV * K +...
                            k * numV + v)/(total),wb)
                    end
                end
            end
            [~,id] = max(mean(fit,2));
            self.lambda = range(id);
            
            close(wb)
            fprintf('done\n');
        end
        
        function results = perform_ridge(self,data,varargin)
            % performs ridge regression and returns a structure with the
            % following fields
            %  - Beta
            %  - RSS
            %  - F_statistic
            %  - P_value
            %
            % The dimension of each field corresponds to the dimensions of
            % the data. Beta is a cell structure with each cell being
            % a column vector of length p (number of predictors).
            %
            % required input is
            %  - data  : a matrix of empirically observed BOLD timecourses
            %            whose columns correspond to time (volumes).
            %
            % optional inputs are
            %  - lambda: shrinkage parameter
            %  - mask  : mask file for selecting voxels
            
            text = 'performing ridge regression';
            fprintf('%s with lambda = %.2f...\n',text, self.lambda)
            wb = waitbar(0,sprintf('%s...',text),'Name',self.is);
            
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