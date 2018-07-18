classdef IRM < handle
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
    % Input-referred model (IRM) mapping tool.
    %
    % irm = IRM(params) creates an instance of the IRM class.
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
    %   - hrf = IRM.get_hrf();
    %   - stimulus = IRM.get_stimulus();
    %   - tc = IRM.get_timecourses();
    %   - IRM.set_hrf(hrf);
    %   - IRM.set_stimulus(stimulus);
    %   - IRM.create_timecourses();
    %   - results = IRM.mapping(data);
    %
    % use help IRM.function to get more detailed help on any specific function
    % (e.g. help IRM.mapping)
    %
    % typical workflow:
    % 1. irm = IRM(params);
    % 2. irm.set_stimulus();
    % 3. irm.create_timecourse(FUN,xdata);
    % 4. results = irm.mapping(data);
    
    properties (Access = private)
        
        is
        
        % functions
        two_gamma               % two gamma hrf function
        
        % parameters
        f_sampling
        p_sampling
        n_samples
        n_rows
        n_cols
        n_slices
        n_total
        n_predictors
        n_points
        idx
        l_hrf
        hrf
        stimulus
        xdata
        tc_fft
    end
    
    methods (Access = public)
        
        function self = IRM(params,varargin)
            % constructor
            p = inputParser;
            addRequired(p,'params',@isstruct);
            addOptional(p,'hrf',[]);
            p.parse(params,varargin{:});
            
            self.is = 'input-referred modeling tool';
            
            self.two_gamma = @(t) (6*t.^5.*exp(-t))./gamma(6)...
                -1/6*(16*t.^15.*exp(-t))/gamma(16);
            
            self.f_sampling = p.Results.params.f_sampling;
            self.p_sampling = 1/self.f_sampling;
            self.n_samples = p.Results.params.n_samples;
            self.n_rows = p.Results.params.n_rows;
            self.n_cols = p.Results.params.n_cols;
            self.n_slices = p.Results.params.n_slices;
            self.n_total = self.n_rows*self.n_cols*self.n_slices;
            
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
            % If a single hrf is used for every voxel, this function 
            % returns a column vector.
            % If a unique hrf is used for each voxel, this function returns 
            % a matrix with columns corresponding to time and the remaining
            % dimensions reflecting the spatial dimensions of the data.
            if size(self.hrf,2)>1
                hrf = reshape(self.hrf,self.l_hrf,...
                    self.n_rows,self.n_cols,self.n_slices);
            else
                hrf = self.hrf;
            end
        end
        
        function stimulus = get_stimulus(self)
            % returns the stimulus used by the class as a column vector
            stimulus = self.stimulus;
        end
        
        function tc = get_timecourses(self)
            % returns the timecourses predicted based on the stimulus 
            % protocol and each combination of the input-referred model 
            % parameters as a time-by-combinations matrix.
            % Note that the predicted timecourses have not been convolved 
            % with a hemodynamic response.
            tc = ifft(self.tc_fft);
        end
        
        function set_hrf(self,hrf)
            % replace the hemodynamic response with a new hrf column vector
            % or a matrix whose columns correspond to time.
            % The remaining dimensionsneed to match those of the data.
            self.l_hrf = size(hrf,1);
            if ndims(hrf)==4
                self.hrf = reshape(hrf,self.l_hrf,self.n_total);
            else
                self.hrf = hrf;
            end
        end
        
        function set_stimulus(self,stimulus)
            % provide a stimulus column vector.
            self.stimulus = stimulus;
        end
        
        function create_timecourse(self,FUN,xdata)
            % creates predicted timecourses based on the stimulus protocol 
            % and a range of parameters for an input referred model.
            %
            % required inputs are
            % - FUN  : a function handle defining the input referred model
            % - xdata: an n-by-p matrix defining the parameter space with 
            %          n values in p parameters
            text = 'creating timecourses...';
            fprintf('%s\n',text)
            wb = waitbar(0,text,'Name',self.is);
            
            self.xdata = xdata;
            [n_observations,self.n_predictors] = size(xdata);
            self.n_points = n_observations^self.n_predictors;
            i = (0:self.n_points-1)';
            self.idx = zeros(self.n_points,self.n_predictors);
            
            for p=1:self.n_predictors
                self.idx(:,p) = mod(floor(i/(n_observations^(self.n_predictors-p))),...
                    n_observations) + 1;
            end
            
            tc = zeros(self.n_samples,self.n_points);
            x = zeros(self.n_predictors,1);
            for j=1:self.n_points
                
                for p=1:self.n_predictors
                    x(p) = xdata(self.idx(j,p),p);
                end
                tc(:,j) = FUN(self.stimulus,x);
                waitbar(j/self.n_points,wb);
            end
            self.tc_fft = fft(tc);
            close(wb)
            fprintf('done\n');
        end
        
        function results = mapping(self,data,varargin)
            % identifies the best fitting timecourse for each voxel and
            % returns the corresponding parameter values of the
            % input-referred model. The class returns a structure with two
            % fields.
            %  - R: correlations (fit) - dimension corresponds to the 
            %       dimensions of the data.
            %  - P: estimate parameters - dimension corresponds to the 
            %       dimensions of the data + 1.
            %
            % required inputs are
            %  - data  : a matrix of empirically observed BOLD timecourses
            %            whose columns correspond to time (volumes).
            %
            % optional inputs are
            %  - threshold: minimum voxel intensity (default = 100.0)
            
            text = 'mapping input-referred model...';
            fprintf('%s\n',text)
            wb = waitbar(0,text,'Name',self.is);
            
            p = inputParser;
            addRequired(p,'data',@isnumeric);
            addOptional(p,'threshold',100);
            p.parse(data,varargin{:});
            
            data = p.Results.data;
            threshold = p.Results.threshold;
            
            data = reshape(data(1:self.n_samples,:,:,:),...
                self.n_samples,self.n_total);
            mean_signal = mean(data);
            data = zscore(data);
            mag_d = sqrt(sum(data.^2));
            
            results.R = zeros(self.n_total,1);
            results.P = zeros(self.n_total,self.n_predictors);
            
            if size(self.hrf,2)==1
                hrf_fft = fft(repmat([self.hrf;...
                    zeros(self.n_samples-self.l_hrf,1)],...
                    [1,self.n_points]));
                tc = zscore(ifft(self.tc_fft.*hrf_fft))';
                
                mag_tc = sqrt(sum(tc.^2,2));
                for v=1:self.n_total
                    if mean_signal(v)>threshold
                        CS = (tc*data(:,v))./...
                            (mag_tc*mag_d(v));
                        id = isinf(CS) | isnan(CS);
                        CS(id) = 0;
                        [results.R(v),j] = max(CS);
                        for p=1:self.n_predictors
                            results.P(v,p) = self.xdata(self.idx(j,p),p);
                        end
                    end
                    waitbar(v/self.n_total,wb)
                end
            else
                hrf_fft_all = fft([self.hrf;...
                    zeros(self.n_samples-self.l_hrf,self.n_total)]);
                for v=1:self.n_total
                    if mean_signal(v)>threshold
                        hrf_fft = repmat(hrf_fft_all(:,v),...
                            [1,self.n_points]);
                        tc = zscore(ifft(self.tc_fft.*hrf_fft))';
                        mag_tc = sqrt(sum(tc.^2,2));
                        
                        CS = (tc*data(:,v))./...
                            (mag_tc*mag_d(v));
                        id = isinf(CS) | isnan(CS);
                        CS(id) = 0;
                        [results.R(v),j] = max(CS);
                        for p=1:self.n_predictors
                            results.P(v,p) = self.xdata(self.idx(j,p),p);
                        end
                    end
                    waitbar(v/self.n_total,wb)
                end
            end
            results.R = reshape(results.R,self.n_rows,self.n_cols,self.n_slices);
            results.P = squeeze(...
                reshape(...
                results.P,self.n_rows,self.n_cols,self.n_slices,self.n_predictors));
            close(wb)
            fprintf('done\n');
        end
    end
end