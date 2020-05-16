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
    %   - r_stimulus: width & height of stimulus images in pixels
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
        r_stimulus
        idx
        ecc
        pa
        slope
        hrf
        stimulus
        tc_fft
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
            self.r_stimulus = p.Results.parameters.r_stimulus;
            
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
            % a matrix with rows corresponding to time and the remaining
            % dimensions reflecting the spatial dimensions of the data.
            if size(self.hrf,2)>1
                hrf = reshape(self.hrf,self.l_hrf,...
                    self.n_rows,self.n_cols,self.n_slices);
            else
                hrf = self.hrf;
            end
        end
        
        function stimulus = get_stimulus(self)
            % returns the stimulus used by the class as a 3D matrix of
            % dimensions height-by-width-by-time.
            stimulus = reshape(self.stimulus,self.r_stimulus,...
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
            % or a matrix whose rows correspond to time.
            % The remaining dimensionsneed to match those of the data.
            self.l_hrf = size(hrf,1);
            if ndims(hrf)>2
                self.hrf = reshape(hrf,self.l_hrf,self.n_total);
            else
                self.hrf = hrf;
            end
        end
        
        function set_stimulus(self,stimulus)
            % provide a stimulus matrix.
            % This is useful if the .png files have already been imported
            % and stored in matrix form.
            % Note that the provided stimulus matrix can be either 3D
            % (height-by-width-by-time) or 2D (height*width-by-time).
            if ndims(stimulus)==3
                self.stimulus = reshape(stimulus,...
                    self.r_stimulus^2,...
                    self.n_samples);
            else
                self.stimulus = stimulus;
            end
        end
        
        function import_stimulus(self)
            % imports a series of .png files constituting the stimulation
            % protocol of the pRF experiment.
            % This series is stored internally in matrix format
            % (height-by-width-by-time).
            % The stimulus is required to generate timecourses.
            
            [~,path] = uigetfile('*.png',...
                'Please select the first .png file');
            files = dir([path,'*.png']);
            
            im = imread([path,files(1).name]);
            self.stimulus = zeros(self.r_stimulus,self.r_stimulus,...
                self.n_samples);
            self.stimulus(:,:,1) = im(:,:,1);
            l = regexp(files(1).name,'\d')-1;
            prefix = files(1).name(1:l);
            name = {files.name};
            str  = sprintf('%s#', name{:});
            num  = sscanf(str, [prefix,'%d.png#']);
            [~, index] = sort(num);
            
            for t=2:self.n_samples
                im = imread([path,files(index(t)).name]);
                self.stimulus(:,:,t) = im(:,:,1);
            end
            mn = min(self.stimulus(:));
            range = max(self.stimulus(:))-mn;
            
            self.stimulus = (reshape(self.stimulus,...
                self.r_stimulus^2,...
                self.n_samples)-mn)/range;
        end
        
        function create_timecourses(self,varargin)
            % creates predicted timecourses based on the effective stimulus
            % and a range of isotropic receptive fields.
            % Isotropic receptive fields are generated for a grid of
            % location (x,y) and size parameters.
            %
            % optional inputs are
            %  - num_xy      : steps in x and y direction      (default =  30.0)
            %  - max_radius  : radius of the field of fiew     (default =  10.0)
            %  - num_slope   : steps from lower to upper bound (default =  10.0)
            %  - min_slope   : lower bound of RF size slope    (default =   0.1)
            %  - max_slope   : upper bound of RF size slope    (default =   1.2)
            %  - css_exponent: compressive spatial summation   (default =   1.0)
            %  - sampling    : eccentricity sampling type      (default = 'log')
            
            progress('creating timecourses');
            
            p = inputParser;
            addOptional(p,'num_xy',30);
            addOptional(p,'max_radius',10);
            addOptional(p,'num_slope',10);
            addOptional(p,'min_slope',0.1);
            addOptional(p,'max_slope',1.2);
            addOptional(p,'css_exponent',1);
            addOptional(p,'sampling','log');
            p.parse(varargin{:});
            
            num_xy = p.Results.num_xy;
            max_radius = p.Results.max_radius;
            num_slope = p.Results.num_slope;
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
                sigma = self.ecc(self.idx(j,2)) * self.slope(self.idx(j,3));
                W(j,:) = self.gauss(x,y,sigma,X_,Y_)';
                progress(j / self.n_points * 19)
            end
            
            tc = (W * self.stimulus).^css_exponent;
            progress(20)
            self.tc_fft = fft(tc');
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
            %  - data  : a matrix of empirically observed BOLD timecourses
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
            
            data = single(p.Results.data);
            threshold = p.Results.threshold;
            mask = p.Results.mask;
            
            data = reshape(data(1:self.n_samples,:,:,:),...
                self.n_samples,self.n_total);
            mean_signal = mean(data);
            
            
            if isempty(mask)
                mask = mean_signal>=threshold;
            end
            mask = mask(:);
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
                    zeros(self.n_samples-self.l_hrf,1)],...
                    [1,self.n_points]));
                tc = zscore(ifft(self.tc_fft.*hrf_fft))';
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
                    
                    progress((v-1) / n_voxels * 20)
                end
            else
                hrf_fft_all = fft([self.hrf(:,mask);...
                    zeros(self.n_samples-self.l_hrf,n_voxels)]);
                for m=1:n_voxels
                    v = voxel_index(m);
                    
                    tc = zscore(ifft(self.tc_fft.*hrf_fft_all(:,m)))';
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
                    
                    progress((v-1) / n_voxels * 20)
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
end
