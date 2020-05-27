classdef HGR < handle
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % %%%                               LICENSE                             %%%
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    % Copyright 2020 Salil Bhat & Mario Senden
    %
    % This program is free software: you can redistribute it and/or modify
    % it under the terms of the GNU Lesser General Public License as tpublished
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
    %
    % Hashed Gaussian Regression (HGR) tool.
    %
    % hgr = HGR(parameters) creates an instance of the HGR class.
    %
    % parameters is a structure with 7 required fields
    %   - f_sampling : sampling frequency (1/TR)
    %   - r_stimulus : width & height of stimulus images in pixels
    %   - n_features : number of features (hashed Gaussians)
    %   - n_gaussians: number of Gaussians per feature
    %   - n_voxels   : total number of voxels in data
    %   - fwhm       : full width at half maximum of Gaussians
    %   - eta        : learning rate (inverse of regularization parameter)
    %
    % optional inputs are
    %   - l_kernel   : length of convolution kernel (two-gamma hresults)
    %
    % this class has the following functions
    %
    %   - [mask, corr_fit] = HGR.get_best_voxels(data, stimulus);
    %   - gamma = HGR.get_features();
    %   - results = HGR.get_parameters();
    %   - theta = HGR.get_weights();
    %   - tc = HGR.get_timecourses();
    %   - HGR.set_parameters(parameters);
    %   - HGR.reset();
    %   - HGR.ridge(data, stimulus);
    %   - HGR.update(data, stimulus);
    %
    % use help HGR.function to get more detailed help on any specific function
    % (e.g. help HGR.ridge)
    %
    % typical offline workflow:
    % 1. hgr = HGR(parameters);
    % 2. hgr.ridge(data, stimulus);
    % 3. hgr.get_parameters();
    
    properties (Access = private)

        % functions
        gauss            % 2D Gaussian function
        two_gamma        % two gamma hresults function
        
        % class objects
        data_processor 
        phi_processor

        % parameters
        p_sampling       % sampling rate
        r_stimulus       % stimulus width & height
        n_pixels         % total number of pixels
        n_features       % number of features
        n_gaussians      % number of gaussians per feature
        n_voxels         % number of voxels
        fwhm             % fwhm of each gaussian
        eta              % learning rate (gradient descend)
        lambda           % penalty parameter (ridge)

        % variables
        theta            % feature weights
        gamma            % features (hashed Gaussians)

        l_hrf            % length of hrf
        hrf              % hrf convolution kernel

    end
    methods (Access = public)

        function self = HGR(parameters, varargin)

            % constructor
            p = inputParser;
            addRequired(p,'parameters',@isstruct);
            addOptional(p,'l_kernel',34);
            p.parse(parameters,varargin{:});

            parameters = p.Results.parameters;
            self.l_hrf = p.Results.l_kernel;
            self.two_gamma = @(t) (6*t.^5.*exp(-t))./gamma(6)...
                -1/6*(16*t.^15.*exp(-t))/gamma(16);
            self.gauss = @(mu_x,mu_y,sigma,X,Y) exp(-((X - mu_x).^2 +...
                (Y - mu_y).^2) ./ (2 * sigma.^2));
            
            
            self.p_sampling = 1 / parameters.f_sampling;
            self.r_stimulus = parameters.r_stimulus;
            self.n_pixels = self.r_stimulus^2;
            self.n_features = parameters.n_features;
            self.n_gaussians = parameters.n_gaussians;
            self.n_voxels = parameters.n_voxels;
            self.fwhm = parameters.fwhm * self.r_stimulus;
            self.eta = parameters.eta / self.n_features;
            self.lambda = 1 / self.eta;
            self.create_gamma();
            self.theta = zeros(self.n_features,self.n_voxels);
            self.hrf = self.two_gamma(0:self.p_sampling:...
                self.l_hrf - 1)';

            self.data_processor = online_processor(...
                self.n_voxels,...
                'sampling_rate', self.p_sampling,...
                'l_kernel', self.l_hrf);
            
            self.phi_processor = online_processor(...
                self.n_features,...
                'sampling_rate', self.p_sampling,...
                'l_kernel', self.l_hrf);
            
        end

        function update(self,data,stimulus)
            % performs a single gradient descent update based on current
            % time point's data and stimulus. Online convolution and
            % z-normalization is handled internally.  
            %
            % required inputs are
            %  - data    : a row vector of observed BOLD activation levels 
            %              per voxel.
            %  - stimulus: a row vector of pixel intensities.
            phi = stimulus * self.gamma;
            phi = self.phi_processor.convolve(phi);
            phi = self.phi_processor.update(phi);
            y = self.data_processor.update(data);
            self.theta = self.theta + self.eta *...
                (phi' * y - phi' * phi * self.theta);
        end

        function ridge(self,data,stimulus)
            % performs ridge regression with stimulus encoded by hashed
            % Gaussians as predictors.
            %
            % required inputs are
            %  - data    : a matrix of empirically observed BOLD timecourses
            %              whose rows correspond to time (volumes).
            %  - stimulus: a time by number of pixels stimulus matrix.
            I = eye(self.n_features) * self.lambda;
            phi = zscore(self.convolution(stimulus * self.gamma));
            self.theta = (phi' * phi + I) \ phi' * zscore(data);
        end

        function gamma = get_features(self)
            % returns hashed Gaussians as a number of pixels by number of
            % features matrix.
            gamma = self.gamma;
        end

        function theta = get_weights(self)
            % returns learned regression weights as a number of features by
            % number of voxels matrix.
            theta = self.theta;
        end

        function results = get_parameters(self,varargin)
            % estimates population receptive field (2D Gaussian) parameters
            % based on raw receptive fields given by features and their
            % regression weights.
            %
            % returns a structure with the following fields
            %  - corr_fit
            %  - mu_x
            %  - mu_y
            %  - sigma
            %  - eccentricity
            %  - polar_angle
            %
            % each field is a column vector with number of voxels elements
            %
            % optional inputs are
            %  - n_batch   : batch size                       (default = 10000)
            %  - max_radius: radius of measured visual field  (default =    10)
            %  - alpha     : shrinkage parameter              (default =     1)
            %  - mask      : binary mask for selecting voxels
            
            progress('estimating pRF parameters')
            
            p = inputParser;
            addOptional(p,'mask',true(self.n_voxels,1));
            addOptional(p,'n_batch',10000);
            addOptional(p,'max_radius',10);
            addOptional(p,'alpha',1);
            p.parse(varargin{:});
            msk = p.Results.mask;
            n_batch = p.Results.n_batch;
            max_radius = p.Results.max_radius;
            alpha = p.Results.alpha;


            idx = (1:self.n_voxels)';
            idx = idx(msk);
            n_msk = sum(msk);

            results.mu_x = nan(self.n_voxels,1);
            results.mu_y = nan(self.n_voxels,1);
            results.sigma = nan(self.n_voxels,1);

            Y = linspace(max_radius, -max_radius, self.r_stimulus)' * ones(1,self.r_stimulus);
            X = ones(self.r_stimulus, 1) * linspace(-max_radius, max_radius, self.r_stimulus);
            X = X(:);
            Y = Y(:);

            s = linspace(1e-3, max_radius, 25);
            r = linspace(0, sqrt(2 * max_radius^2),25);
            [S,R] = meshgrid(s,r);
            S = S(:);
            R = R(:);


            I = zeros(numel(S),1);

            for i=1:numel(S)
                x = cos(pi/4) * R(i);
                y = sin(pi/4) * R(i);
                I(i) = mean(self.gauss(x, y, S(i), X, Y));
            end
            P = [I, R];
            beta = (P' * P) \ P' * S;

            for v=0:n_batch:n_msk-n_batch
                progress(v / n_msk * 20)
                batch = idx(v+1:v+n_batch);
                im = self.gamma * self.theta(:,batch);
                [mx,pos] = max(im);
                mn = min(im);
                range = mx - mn;
                im = ((im - mn) ./ range).^alpha;
                m_image = mean(im)';
                cx = floor((pos-1) / self.r_stimulus);
                cy = mod(pos-1, self.r_stimulus);
                results.mu_x(batch) = cx / self.r_stimulus * max_radius * 2 - max_radius;
                results.mu_y(batch) = -(cy / self.r_stimulus * max_radius * 2 - max_radius);

                results.sigma(batch) = [m_image, sqrt(results.mu_x(batch).^2 +...
                    results.mu_y(batch).^2)] * beta;
               
                
            end
           
            if isempty(v)
                batch = idx; 
            else
                batch = idx(v+1:end);
            end
            progress(20)
            im = self.gamma * self.theta(:,batch);
            [mx,pos] = max(im);
            mn = min(im);
            range = mx - mn;
            im = ((im - mn) ./ range).^alpha;
            m_image = mean(im)';
            cx = floor((pos-1) / self.r_stimulus);
            cy = mod(pos-1, self.r_stimulus);
            results.mu_x(batch) = cx / self.r_stimulus * max_radius * 2 - max_radius;
            results.mu_y(batch) = -(cy / self.r_stimulus * max_radius * 2 - max_radius);
            results.sigma(batch) = [m_image, sqrt(results.mu_x(batch).^2 +...
                results.mu_y(batch).^2)] * beta;
            results.polar_angle = angle(results.mu_x + results.mu_y * 1i);
            results.eccentricity = abs(results.mu_x + results.mu_y * 1i);

        end

        function tc = get_timecourses(self, stimulus)
            % returns the timecourses predicted based on an encoding of the
            % provided stimulus and the feature weights.
            phi = zscore(self.convolution(stimulus * self.gamma));
            tc = phi * self.theta;
        end

        function set_parameters(self,parameters)
            % change parameters of the class
            %
            % required input
            %  - parameters: a structure containing all parameters required
            %                required by the class
            
            self.r_stimulus = parameters.r_stimulus;
            self.n_pixels = self.r_stimulus^2;
            self.n_features = parameters.n_features;
            self.n_gaussians = parameters.n_gaussians;
            self.n_voxels = parameters.n_voxels;
            self.fwhm = parameters.fwhm;
            self.eta = parameters.eta / self.n_features;
            self.lambda = 1 / self.eta;
            self.create_gamma();
        end

        function reset(self)
            % reset all internal states of the class
            %             
            % use this function prior to conducting real time estimation
            % for a new set of data
            self.theta = zeros(self.n_features,self.n_voxels);
            self.phi_processor.reset();
            self.data_processor.reset();
        end

        function [mask,corr_fit] = get_best_voxels(self,data,stimulus,varargin)
            % uses blocked cross-validation to obtain the best fitting
            % voxels and returns a mask as well the correlation fit per
            % voxel
            %
            % required inputs are
            %  - data    : a matrix of empirically observed BOLD timecourses
            %              whose rows correspond to time (volumes).
            %  - stimulus: a time by number of pixels stimulus matrix.
            %
            % optional inputs are
            %  - type    : cutoff type           (default = 'percentile')
            %  - cutoff  : cutoff value          (default = 95)
            %  - n_splits: number of data splits (default = 4)
            
            p = inputParser;
            addRequired(p,'data');
            addRequired(p,'stimulus');
            addOptional(p,'type','percentile');
            addOptional(p,'cutoff', 95);
            addOptional(p,'n_splits', 4);

            p.parse(data, stimulus, varargin{:});
            data = p.Results.data;
            stimulus = p.Results.stimulus;
            type = p.Results.type;
            cutoff = p.Results.cutoff;
            n_splits = p.Results.n_splits;

            n_time = size(data, 1);
            n_steps = n_splits - 1;
            n_samples = floor(n_time / n_splits);

            corr_fit = zeros(self.n_voxels, n_splits);
            for i = 1:n_steps
                bound = i * n_samples;
                train_data = zscore(data(1:bound,:));
                train_stim = stimulus(1:bound,:);
                test_data = zscore(data(bound+1:end,:));
                test_stim = stimulus(bound+1:end,:);

                self.ridge(train_data,train_stim);
                Y = self.get_timecourses(test_stim);
                mag_Y = sqrt(sum(Y.^2));
                mag_data = sqrt(sum(test_data.^2));
                corr_fit(:,i) = (sum(Y .* test_data) ./ (mag_Y .* mag_data))';
            end
            corr_fit = mean(corr_fit, 2);

            if strcmp(type,'percentile')
                threshold = prctile(corr_fit, cutoff);
                mask = corr_fit>=threshold;
            elseif strcmp(type,'threshold')
                mask = corr_fit>=cutoff;
            elseif strcmp(type,'number')
                corr_fit(isnan(corr_fit)) = -1;
                [val] = sort(corr_fit, 'descend');
                threshold = val(cutoff);
                mask = corr_fit>=threshold;
            else
               error('Wrong type. Choose either ''percentile'', ''number'' or ''threshold''')
            end
        end

    end

    methods (Access = private)
        function create_gamma(self)
            Y = linspace(0,self.r_stimulus,self.r_stimulus)' * ones(1,self.r_stimulus);
            X = ones(self.r_stimulus,1) * linspace(0,self.r_stimulus,self.r_stimulus);
            sigma = self.fwhm / ( 2 * sqrt(2 * log(2)));
            self.gamma = zeros(self.r_stimulus,self.r_stimulus,self.n_features);
            pix_id = linspace(0, self.n_pixels-1,...
                self.n_features * self.n_gaussians);
            order = randperm(self.n_features * self.n_gaussians);
            pix_id = pix_id(order);
            x = floor(pix_id/self.r_stimulus) + 1;
            y = mod(pix_id,self.r_stimulus) + 1;
            for i=0:self.n_features-1
                for j=1:self.n_gaussians

                    self.gamma(:,:,i+1) = self.gamma(:,:,i+1) +...
                        self.gauss(x(i*self.n_gaussians+j),...
                        y(i*self.n_gaussians+j),sigma,X,Y);
                end
                self.gamma(:,:,i+1) = self.gamma(:,:,i+1) /...
                    sum(sum(self.gamma(:,:,i+1)));
            end

            self.gamma = (reshape(self.gamma,self.n_pixels,self.n_features));
        end

        function x_conv = convolution(self, x)
            n_samples = size(x, 1);
            kernel = [self.hrf; zeros(n_samples, 1)];
            x = [x; zeros(ceil(self.l_hrf / self.p_sampling), self.n_features)];
            x_conv = ifft(fft(x) .* fft(kernel));
            x_conv = x_conv(1:n_samples, :);
        end

    end

end
