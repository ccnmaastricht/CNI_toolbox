classdef pRF < handle
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
    % Population receptive field (pRF) mapping tool.
    %
    % prf = pRF(parameters) creates an instance of the pRF class.
    %
    % parameters is a structure with 7 required fields
    %   - f_sampling: sampling frequency (1/TR)
    %   - n_samples : number of samples (volumes)
    %   - n_rows    : number of rows (in-plane resolution)
    %   - n_cols    : number of columns (in-plance resolution)
    %   - n_slices  : number of slices
    %   - h_stimulus: height of stimulus images in pixels
    %   - w_stimulus: width of stimulus images in pixels
    %
    % optional inputs are
    %   - hrf       : either a column vector containing a single hemodynamic
    %                 response used for every voxel;
    %                 or a tensor with a unique hemodynamic response along
    %                 its columns for each voxel.
    %                 By default the canonical two-gamma hemodynamic response
    %                 function is generated internally based on provided parameters.
    %
    % this class has the following functions
    %
    %   - hrf = pRF.get_hrf();
    %   - stimulus = pRF.get_stimulus();
    %   - tc = pRF.get_timecourses();
    %   - pRF.set_hrf(hrf);
    %   - pRF.set_stimulus(stimulus);
    %   - pRF.import_stimulus();
    %   - pRF.create_timecourses();
    %   - results = pRF.mapping(data);
    %
    % use help pRF.function to get more detailed help on any specific function
    % (e.g. help pRF.mapping)
    %
    % typical workflow:
    % 1. prf = pRF(parameters);
    % 2. prf.import_stimulus();
    % 3. prf.create_timecourses();
    % 4. results = prf.mapping(data);
    
    properties (Access = private)
        
        % functions
        two_gamma               % two gamma hrf function
        gauss                   % isotropic Gaussian
        
        % parameters
        f_sampling
        p_sampling
        n_samples
        n_points
        n_rows
        n_cols
        n_slices
        n_total
        l_hrf
        h_stimulus
        w_stimulus
        r_stimulus
        use_slope
        idx
        ecc
        pa
        slope
        sigma
        hrf
        stimulus
        tc_fft
    end
    
    methods (Access = private)
        function padded_stimulus = zeropad(self, stimulus)
            padded_stimulus = zeros(self.r_stimulus,...
                self.r_stimulus, self.n_samples);
            
            width_half = floor((self.r_stimulus - self.w_stimulus) / 2);
            width_lower = width_half + 1;
            width_upper = width_half + self.w_stimulus;
            
            height_half = floor((self.r_stimulus - self.h_stimulus) / 2);
            height_lower = height_half + 1;
            height_upper = height_half + self.h_stimulus;
            
            padded_stimulus(height_lower:height_upper,...
                width_lower:width_upper,:) = stimulus;
        end
        
        function create_timecourses_slope(self, p)
            % creates predicted timecourses based on the effective stimulus
            % and a range of isotropic receptive fields.
            % Isotropic receptive fields are generated for a grid of
            % location (x,y) and slope parameters.
        
            progress('creating timecourses');
            
            num_xy = p.Results.num_xy;
            max_radius = p.Results.max_radius;
            num_slope = p.Results.num_size;
            min_slope = p.Results.min_slope;
            max_slope = p.Results.max_slope;
            css_exponent = p.Results.css_exponent;
            sampling = p.Results.sampling;
            self.n_points = num_xy^2 * num_slope;
            
            X_ = ones(self.r_stimulus,1) * linspace(-max_radius,...
                max_radius,self.r_stimulus);
            Y_ = -linspace(-max_radius,max_radius,...
                self.r_stimulus)' * ones(1,self.r_stimulus);
            
            X_ = X_(:);
            Y_ = Y_(:);
            
            i = (0:self.n_points-1)';
            self.idx = [floor(i/(num_xy*num_slope))+1,...
                mod(floor(i/(num_slope)),num_xy)+1,...
                mod(i,num_slope)+1];
            
            if strcmp(sampling,'log')
                self.ecc = exp(linspace(log(.1),log(max_radius),num_xy));
            elseif strcmp(sampling,'linear')
                self.ecc = linspace(.1,max_radius,num_xy);
            end
            
            self.pa = linspace(0,(num_xy-1)/num_xy*2*pi,num_xy);
            self.slope = linspace(min_slope,max_slope,num_slope);
            
            W = single(zeros(self.n_points,...
                self.r_stimulus^2));
            
            
            for j=1:self.n_points
                x = cos(self.pa(self.idx(j,1))) * self.ecc(self.idx(j,2));
                y = sin(self.pa(self.idx(j,1))) * self.ecc(self.idx(j,2));
                sigm = self.ecc(self.idx(j,2)) * self.slope(self.idx(j,3));
                W(j,:) = self.gauss(x,y,sigm,X_,Y_)';
                progress(j / self.n_points * 19)
            end
            
            tc = (W * self.stimulus).^css_exponent;
            progress(20)
            sdev_tc = std(tc,[],2);
            idx_remove = sdev_tc==0;
            num_remove = sum(idx_remove);
            self.n_points = self.n_points - num_remove;
            tc(idx_remove, :) = [];
            self.idx(idx_remove, :) = [];
            self.tc_fft = fft(tc');
        end
        
        function create_timecourses_sigma(self,p)
            % creates predicted timecourses based on the effective stimulus
            % and a range of isotropic receptive fields.
            % Isotropic receptive fields are generated for a grid of
            % location (x,y) and size parameters.
            %
            
            progress('creating timecourses');
          
            num_xy = p.Results.num_xy;
            max_radius = p.Results.max_radius;
            num_sigma = p.Results.num_size;
            min_sigma = p.Results.min_sigma;
            max_sigma = p.Results.max_sigma;
            css_exponent = p.Results.css_exponent;
            sampling = p.Results.sampling;
            self.n_points = num_xy^2 * num_sigma;
            
            X_ = ones(self.r_stimulus,1) * linspace(-max_radius,...
                max_radius,self.r_stimulus);
            Y_ = -linspace(-max_radius,max_radius,...
                self.r_stimulus)' * ones(1,self.r_stimulus);
            
            X_ = X_(:);
            Y_ = Y_(:);
            
            i = (0:self.n_points-1)';
            self.idx = [floor(i/(num_xy*num_sigma))+1,...
                mod(floor(i/(num_sigma)),num_xy)+1,...
                mod(i,num_sigma)+1];
            
            if strcmp(sampling,'log')
                self.ecc = exp(linspace(log(.1),log(max_radius),num_xy));
            elseif strcmp(sampling,'linear')
                self.ecc = linspace(.1,max_radius,num_xy);
            end
            
            self.pa = linspace(0,(num_xy-1)/num_xy*2*pi,num_xy);
            self.sigma = linspace(min_sigma, max_sigma * max_radius, num_sigma);
            
            W = single(zeros(self.n_points,...
                self.r_stimulus^2));
            
            
            for j=1:self.n_points
                x = cos(self.pa(self.idx(j,1))) * self.ecc(self.idx(j,2));
                y = sin(self.pa(self.idx(j,1))) * self.ecc(self.idx(j,2));
                W(j,:) = self.gauss(x,y,self.sigma(self.idx(j,3)),X_,Y_)';
                progress(j / self.n_points * 19)
            end
            
            tc = (W * self.stimulus).^css_exponent;
            progress(20)
            sdev_tc = std(tc,[],2);
            idx_remove = sdev_tc==0;
            num_remove = sum(idx_remove);
            self.n_points = self.n_points - num_remove;
            tc(idx_remove, :) = [];
            self.idx(idx_remove, :) = [];
            self.tc_fft = fft(tc');
        end
        
        function results = mapping_slope(self, p)
            
            progress('mapping population receptive fields')
            
            data = single(p.Results.data);
            threshold = p.Results.threshold;
            mask = p.Results.mask;
            
            data = reshape(data(1:self.n_samples,:,:,:),...
                self.n_samples,self.n_total);
            mean_signal = mean(data);
            sdev_signal = std(data);
            
            if isempty(mask)
                mask = mean_signal>=threshold;
            end
            mask = logical(mask(:));
            mask = mask & (sdev_signal(:) > 0);
            voxel_index = find(mask);
            n_voxels = numel(voxel_index);
            
            data = zscore(data(:,mask));
            mag_d = sqrt(sum(data.^2));
            
            results.corr_fit = zeros(self.n_total,1);
            results.mu_x = zeros(self.n_total,1);
            results.mu_y = zeros(self.n_total,1);
            results.sigma = zeros(self.n_total,1);
            
            if size(self.hrf,2)==1
                hrf_fft = fft(repmat([self.hrf;...
                    zeros(self.n_samples,1)],...
                    [1,self.n_points]));
                tc = ifft(self.tc_fft.*hrf_fft);
                tc = zscore(tc(1:self.n_samples, :))';
                mag_tc = sqrt(sum(tc.^2,2));
                for m=1:n_voxels
                    v = voxel_index(m);
                    
                    CS = (tc*data(:,m))./...
                        (mag_tc*mag_d(m));
                    id = isinf(CS) | isnan(CS);
                    CS(id) = 0;
                    [results.corr_fit(v),j] = max(CS);
                    results.mu_x(v) = cos(self.pa(self.idx(j,1))) * ...
                        self.ecc(self.idx(j,2));
                    results.mu_y(v) = sin(self.pa(self.idx(j,1))) * ...
                        self.ecc(self.idx(j,2));
                    results.sigma(v) = self.ecc(self.idx(j,2)) * ...
                        self.slope(self.idx(j,3));
                    
                    progress(m / n_voxels * 20)
                end
            else
                hrf_fft_all = fft([self.hrf(:,mask);...
                    zeros(self.n_samples,n_voxels)]);
                for m=1:n_voxels
                    v = voxel_index(m);
                    
                    tc = ifft(self.tc_fft.*hrf_fft_all(:,m));
                    tc = zscore(tc(1:self.n_samples, :))';
                    mag_tc = sqrt(sum(tc.^2,2));
                    
                    CS = (tc*data(:,m))./...
                        (mag_tc*mag_d(m));
                    id = isinf(CS) | isnan(CS);
                    CS(id) = 0;
                    [results.corr_fit(v),j] = max(CS);
                    results.mu_x(v) = cos(self.pa(self.idx(j,1))) * ...
                        self.ecc(self.idx(j,2));
                    results.mu_y(v) = sin(self.pa(self.idx(j,1))) * ...
                        self.ecc(self.idx(j,2));
                    results.sigma(v) = self.ecc(self.idx(j,2)) * ...
                        self.slope(self.idx(j,3));
                    
                    progress(m / n_voxels * 20)
                end
            end
            results.corr_fit = reshape(results.corr_fit,self.n_rows,self.n_cols,self.n_slices);
            results.mu_x = reshape(results.mu_x,self.n_rows,self.n_cols,self.n_slices);
            results.mu_y = reshape(results.mu_y,self.n_rows,self.n_cols,self.n_slices);
            results.sigma = reshape(results.sigma,self.n_rows,self.n_cols,self.n_slices);
            results.eccentricity = abs(results.mu_x+results.mu_y*1i);
            results.polar_angle = angle(results.mu_x+results.mu_y*1i);
        end
        
        function results = mapping_sigma(self, p)
            
            progress('mapping population receptive fields')
            
            data = single(p.Results.data);
            threshold = p.Results.threshold;
            mask = p.Results.mask;
            
            data = reshape(data(1:self.n_samples,:,:,:),...
                self.n_samples,self.n_total);
            mean_signal = mean(data);
            sdev_signal = std(data);
            
            if isempty(mask)
                mask = mean_signal>=threshold;
            end
            mask = logical(mask(:));
            mask = mask & (sdev_signal(:) > 0);
            voxel_index = find(mask);
            n_voxels = numel(voxel_index);
            
            data = zscore(data(:,mask));
            mag_d = sqrt(sum(data.^2));
            
            results.corr_fit = zeros(self.n_total,1);
            results.mu_x = zeros(self.n_total,1);
            results.mu_y = zeros(self.n_total,1);
            results.sigma = zeros(self.n_total,1);
            
            if size(self.hrf,2)==1
                hrf_fft = fft(repmat([self.hrf;...
                    zeros(self.n_samples,1)],...
                    [1,self.n_points]));
                tc = ifft(self.tc_fft.*hrf_fft);
                tc = zscore(tc(1:self.n_samples, :))';
                mag_tc = sqrt(sum(tc.^2,2));
                for m=1:n_voxels
                    v = voxel_index(m);
                    
                    CS = (tc*data(:,m))./...
                        (mag_tc*mag_d(m));
                    id = isinf(CS) | isnan(CS);
                    CS(id) = 0;
                    [results.corr_fit(v),j] = max(CS);
                    results.mu_x(v) = cos(self.pa(self.idx(j,1))) * ...
                        self.ecc(self.idx(j,2));
                    results.mu_y(v) = sin(self.pa(self.idx(j,1))) * ...
                        self.ecc(self.idx(j,2));
                    results.sigma(v) = self.sigma(self.idx(j,3));
                    
                    progress(m / n_voxels * 20)
                end
            else
                hrf_fft_all = fft([self.hrf(:,mask);...
                    zeros(self.n_samples,n_voxels)]);
                for m=1:n_voxels
                    v = voxel_index(m);
                    
                    tc = ifft(self.tc_fft.*hrf_fft_all(:,m));
                    tc = zscore(tc(1:self.n_samples, :))';
                    mag_tc = sqrt(sum(tc.^2,2));
                    
                    CS = (tc*data(:,m))./...
                        (mag_tc*mag_d(m));
                    id = isinf(CS) | isnan(CS);
                    CS(id) = 0;
                    [results.corr_fit(v),j] = max(CS);
                    results.mu_x(v) = cos(self.pa(self.idx(j,1))) * ...
                        self.ecc(self.idx(j,2));
                    results.mu_y(v) = sin(self.pa(self.idx(j,1))) * ...
                        self.ecc(self.idx(j,2));
                    results.sigma(v) = self.sigma(self.idx(j,3));
                    
                    progress(m / n_voxels * 20)
                end
            end
            results.corr_fit = reshape(results.corr_fit,self.n_rows,self.n_cols,self.n_slices);
            results.mu_x = reshape(results.mu_x,self.n_rows,self.n_cols,self.n_slices);
            results.mu_y = reshape(results.mu_y,self.n_rows,self.n_cols,self.n_slices);
            results.sigma = reshape(results.sigma,self.n_rows,self.n_cols,self.n_slices);
            results.eccentricity = abs(results.mu_x+results.mu_y*1i);
            results.polar_angle = angle(results.mu_x+results.mu_y*1i);
        end
        
    end
    
    
    methods (Access = public)
        
        function self = pRF(parameters,varargin)
            % constructor
            p = inputParser;
            addRequired(p,'parameters',@isstruct);
            addOptional(p,'hrf',[]);
            p.parse(parameters,varargin{:});
            
            self.two_gamma = @(t) (6*t.^5.*exp(-t))./gamma(6)...
                -1/6*(16*t.^15.*exp(-t))/gamma(16);
            self.gauss = @(mu_x,mu_y,sigma,X,Y) exp(-((X-mu_x).^2+...
                (Y-mu_y).^2)/(2*sigma^2));
            
            self.f_sampling = p.Results.parameters.f_sampling;
            self.p_sampling = 1/self.f_sampling;
            self.n_samples = p.Results.parameters.n_samples;
            self.n_rows = p.Results.parameters.n_rows;
            self.n_cols = p.Results.parameters.n_cols;
            self.n_slices = p.Results.parameters.n_slices;
            self.n_total = self.n_rows*self.n_cols*self.n_slices;
            self.h_stimulus = p.Results.parameters.h_stimulus;
            self.w_stimulus = p.Results.parameters.w_stimulus;
            self.r_stimulus = max(self.w_stimulus, self.h_stimulus);
            
            if ~isempty(p.Results.hrf)
                self.l_hrf = size(p.Results.hrf,1);
                if ndims(p.Results.hrf)>2
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
            % a tensor with rows corresponding to time and the remaining
            % dimensions reflecting the spatial dimensions of the data.
            if size(self.hrf,2)>1
                hrf = reshape(self.hrf,self.l_hrf,...
                    self.n_rows,self.n_cols,self.n_slices);
            else
                hrf = self.hrf;
            end
        end
        
        function stimulus = get_stimulus(self)
            % returns the stimulus used by the class as a tensor
            % of rank 3 (height-by-width-by-time).
            stimulus = reshape(self.stimulus(:,1:self.n_samples),...
                self.r_stimulus,...
                self.r_stimulus,self.n_samples);
        end
        
        function tc = get_timecourses(self)
            % returns the timecourses predicted based on the effective
            % stimulus and each combination of pRF model parameters as a
            % time-by-combinations matrix.
            % Note that the predicted timecourses have not been convolved
            % with a hemodynamic response.
            tc = ifft(self.tc_fft);
        end
        
        function set_hrf(self,hrf)
            % replace the hemodynamic response with a new hrf column vector
            % or a tensor whose rows correspond to time.
            % The remaining dimensions need to match those of the data.
            self.l_hrf = size(hrf,1);
            if ndims(hrf)>2
                self.hrf = reshape(hrf,self.l_hrf,self.n_total);
            else
                self.hrf = hrf;
            end
        end
        
        function set_stimulus(self,stimulus)
            % provide a stimulus.
            % This is useful if the .png files have already been imported
            % and stored in matrix form.
            % Note that the provided stimulus tensor can be either rank 3
            % (height-by-width-by-time) or rank 2 (height*width-by-time).
            
            if ndims(stimulus)<3
                stimulus = reshape(stimulus,...
                    self.h_stimulus,...
                    self.w_stimulus,...
                    self.n_samples);
            end
            stimulus = self.zeropad(stimulus);
            self.stimulus = reshape(stimulus,...
                self.r_stimulus^2,...
                self.n_samples);
            
            self.stimulus = [self.stimulus,...
                zeros(self.r_stimulus^2, self.l_hrf)];
        end
        
        function import_stimulus(self)
            % imports a series of .png files constituting the stimulation
            % protocol of the pRF experiment.
            % This series is stored internally in matrix format
            % (height*width-by-time).
            % The stimulus is required to generate timecourses.
            
            [~,path] = uigetfile('*.png',...
                'Please select the first .png file');
            files = dir([path,'*.png']);
            
            im = imread([path,files(1).name]);
            stim = zeros(self.h_stimulus,self.w_stimulus,...
                self.n_samples);
            stim(:,:,1) = im(:,:,1);
            l = regexp(files(1).name,'\d')-1;
            prefix = files(1).name(1:l);
            name = {files.name};
            str  = sprintf('%s#', name{:});
            num  = sscanf(str, [prefix,'%d.png#']);
            [~, index] = sort(num);
            
            for t=2:self.n_samples
                im = imread([path,files(index(t)).name]);
                stim(:,:,t) = im(:,:,1);
            end
            mn = min(stim(:));
            range = max(stim(:))-mn;
            stim = (stim - mn) / range;
            stim = self.zeropad(stim);
            self.stimulus = (reshape(stim,...
                self.r_stimulus^2,...
                self.n_samples)-mn)/range;
            
            self.stimulus = [self.stimulus,...
                zeros(self.r_stimulus^2, self.l_hrf)];
        end
        
        function create_timecourses(self,varargin)
            % creates predicted timecourses based on the effective stimulus
            % and a range of isotropic receptive fields.
            % Isotropic receptive fields are generated for a grid of
            % location (x,y) and either size parameters directly or slopes
            % of the eccentricity-size relationship.
            %
            % optional inputs are
            %  - use_slope   : explore slopes rather than sizes(default =  True)
            %  - num_xy      : steps in x and y direction      (default =  30.0)
            %  - max_radius  : radius of the field of fiew     (default =  10.0)
            %  - num_size    : steps from lower to upper bound (default =  10.0)
            %  - min_slope   : lower bound of RF size slope    (default =   0.1)
            %  - max_slope   : upper bound of RF size slope    (default =   1.2)
            %  - min_sigma   : lower bound of RF size          (default =   0.1)
            %  - max_sigma   : upper bound of RF size          (default =   1.0)
            %  - css_exponent: compressive spatial summation   (default =   1.0)
            %  - sampling    : eccentricity sampling type
            %                  > 'log' (default)
            %                  > 'linear'
            
            progress('creating timecourses');
            
            p = inputParser;
            addOptional(p,'use_slope', true);
            addOptional(p,'num_xy',30); 
            addOptional(p,'max_radius',10);
            addOptional(p,'num_size',10);
            addOptional(p,'min_slope',0.1);
            addOptional(p,'max_slope',1.2);
            addOptional(p,'min_sigma',0.1);
            addOptional(p,'max_sigma',1);
            addOptional(p,'css_exponent',1);
            addOptional(p,'sampling','log');
            p.parse(varargin{:});
            self.use_slope = p.Results.use_slope;
            
            if self.use_slope
                self.create_timecourses_slope(p);
            else
                self.create_timecourses_sigma(p);
            end
            
        end
        
        function results = mapping(self,data,varargin)
            % identifies the best fitting timecourse for each voxel and
            % returns a structure with the following fields
            %  - corr_fit
            %  - mu_x
            %  - mu_y
            %  - sigma
            %  - eccentricity
            %  - polar_angle
            %
            % the dimension of each field corresponds to the dimensions
            % of the data.
            %
            % required inputs are
            %  - data  : a tensor of empirically observed BOLD timecourses
            %            whose rows correspond to time (volumes).
            %
            % optional inputs are
            %  - threshold: minimum voxel intensity (default = 100.0)
            %  - mask     : binary mask for selecting voxels
            
            progress('mapping population receptive fields')
            
            p = inputParser;
            addRequired(p,'data',@isnumeric);
            addOptional(p,'threshold',100);
            addOptional(p,'mask',[]);
            p.parse(data,varargin{:});
            
            if self.use_slope
                results = self.mapping_slope(p);
            else
                results = self.mapping_sigma(p);
            end
            
        end
    end
end
