classdef HGR < handle
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % %%%                               LICENSE                             %%%
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    % Copyright 2019 Salil Bhat & Mario Senden
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

    properties (Access = private)

        is

        % functions
        gauss            % 2D Gaussian function
        two_gamma        % two gamma hrf function

        % parameters
        r_sampling       % sampling rate
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
        phi              % activity (overlap between gamma and stimulus)

        l_kernel         % length of kernel
        kernel           % convolution kernel
        conv_x           % running convolution

        step             % internal step counter
        mean             % running mean
        previous_mean    % mean at one prior step
        M2               % helper variable for calculating sigma
        sigma            % running standard deviation


    end
    methods (Access = public)

        % constructor
        function self = HGR(parameters, varargin)

            % constructor

            self.is = 'hashed-Gaussian regression tool';

            p = inputParser;
            addRequired(p,'parameters',@isstruct);
            addOptional(p,'l_kernel',34);
            p.parse(parameters,varargin{:});

            parameters = p.Results.parameters;
            self.l_kernel = p.Results.l_kernel;
            self.two_gamma = @(t) (6*t.^5.*exp(-t))./gamma(6)...
                -1/6*(16*t.^15.*exp(-t))/gamma(16);
            self.gauss = @(mu_x,mu_y,sigma,X,Y) exp(-((X - mu_x).^2 +...
                (Y - mu_y).^2) ./ (2 * sigma.^2));

            self.r_sampling = parameters.r_sampling;
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
            self.kernel = self.two_gamma(0:self.r_sampling:...
                self.l_kernel - 1)';
            self.step = 1;
            self.mean = zeros(1,self.n_features);
            self.previous_mean = zeros(1,self.n_features);
            self.M2 = ones(1,self.n_features);
            self.sigma = zeros(1,self.n_features);
        end

        function update(self,data,stimulus)
            phi = stimulus * self.gamma;
            phi_conv = self.convolution_step(phi);
            self.phi = self.zscore_step(phi_conv);
            self.theta = self.theta + self.eta *...
                (self.phi' * data - self.phi' * self.phi * self.theta);
        end

        function ridge(self,data,stimulus)
            I = eye(self.n_features) * self.lambda;
            self.phi = zscore(self.convolution(stimulus * self.gamma));
            self.theta = (self.phi' * self.phi + I) \ self.phi' * data;
        end

        function gamma = get_features(self)
            gamma = self.gamma;
        end

        function theta = get_weights(self)
            theta = self.theta;
        end

        function rf = get_parameters(self,varargin)
            p = inputParser;
            addOptional(p,'mask',true(self.n_voxels,1));
            addOptional(p,'s_batch',10000);
            addOptional(p,'max_radius',10);
            addOptional(p,'alpha',1);
            addOptional(p,'silent',false);
            p.parse(varargin{:});
            msk = p.Results.mask;
            s_batch = p.Results.s_batch;
            max_radius = p.Results.max_radius;
            alpha = p.Results.alpha;
            silent = p.Results.silent;


            idx = (1:self.n_voxels)';
            idx = idx(msk);
            n_msk = sum(msk);

            rf.mu_x = nan(self.n_voxels,1);
            rf.mu_y = nan(self.n_voxels,1);
            rf.sigma = nan(self.n_voxels,1);

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

            if ~silent
                text = 'estimating pRF parameters...';
                wb = waitbar(0,text,'Name',self.is);
            end
            for v=0:s_batch:n_msk-s_batch
                batch = idx(v+1:v+s_batch);
                im = self.gamma * self.theta(:,batch);
                [mx,pos] = max(im);
                mn = min(im);
                range = mx - mn;
                im = ((im - mn) ./ range).^alpha;
                m_image = mean(im)';
                cx = floor((pos-1) / self.r_stimulus);
                cy = mod(pos-1, self.r_stimulus);
                rf.mu_x(batch) = cx / self.r_stimulus * max_radius * 2 - max_radius;
                rf.mu_y(batch) = -(cy / self.r_stimulus * max_radius * 2 - max_radius);

                rf.sigma(batch) = [m_image, sqrt(rf.mu_x(batch).^2 +...
                    rf.mu_y(batch).^2)] * beta;
                if ~silent
                    waitbar(v/n_msk,wb)
                end
            end
            if isempty(v)
                batch = idx;
            else
                batch = idx(v+1:end);
            end
            im = self.gamma * self.theta(:,batch);
            [mx,pos] = max(im);
            mn = min(im);
            range = mx - mn;
            im = ((im - mn) ./ range).^alpha;
            m_image = mean(im)';
            cx = floor((pos-1) / self.r_stimulus);
            cy = mod(pos-1, self.r_stimulus);
            rf.mu_x(batch) = cx / self.r_stimulus * max_radius * 2 - max_radius;
            rf.mu_y(batch) = -(cy / self.r_stimulus * max_radius * 2 - max_radius);
            rf.sigma(batch) = [m_image, sqrt(rf.mu_x(batch).^2 +...
                rf.mu_y(batch).^2)] * beta;
            if ~silent
                close(wb)
            end
        end

        function tc = get_timecourses(self)
            tc = self.phi * self.theta;
        end

        function set_parameters(self,parameters)
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
            self.theta = zeros(self.n_features,self.n_voxels);
            self.step = 1;
            self.mean = zeros(1,self.n_features);
            self.previous_mean = zeros(1,self.n_features);
            self.M2 = ones(1,self.n_features);
            self.sigma = zeros(1,self.n_features);
            self.conv_x = zeros(numel(self.kernel),self.n_features);
        end

        function [index,corr_fit] = get_best_voxels(self,data,stimulus,varargin)

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
                train_data = data(1:bound,:);
                train_stim = stimulus(1:bound,:);
                test_data = zscore(data(bound+1:end,:));
                test_stim = stimulus(bound+1:end,:);

                self.ridge(train_data,train_stim);
                gam = self.get_features;
                thet = self.get_weights;
                ph = test_stim * gam;
                Y = zscore(ph * thet);
                mag_Y = sqrt(sum(Y.^2));
                mag_data = sqrt(sum(test_data.^2));
                corr_fit(:,i) = (sum(Y .* test_data) ./ (mag_Y .* mag_data))';
            end
            corr_fit = mean(corr_fit, 2);

            if strcmp(type,'percentile')
                threshold = prctile(corr_fit, cutoff);
                index = corr_fit>=threshold;
            elseif strcmp(type,'threshold')
                index = corr_fit>=cutoff;
            elseif strcmp(type,'number')
                corr_fit(isnan(corr_fit)) = -1;
                [val] = sort(corr_fit, 'descend');
                threshold = val(cutoff);
                index = corr_fit>=threshold;
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
            kernel = [self.kernel; zeros(n_samples, 1)];
            x = [x; zeros(ceil(self.l_kernel / self.r_sampling), self.n_features)];
            x_conv = ifft(fft(x) .* fft(kernel));
            x_conv = x_conv(1:n_samples, :);
        end

        function x_conv = convolution_step(self, x)
            n_samples = numel(self.kernel) - 1;
            x_fft = fft([x; zeros(n_samples,self.n_features)]);
            self.conv_x = [self.conv_x;zeros(1,self.n_features)];
            self.conv_x(self.step:self.step+rows,:) = ...
                self.conv_x(self.step:self.step+rows,:) + ...
                ifft(x_fft .* self.kernel_fft);
            x_conv = self.conv_x(self.step,:);
        end

        function x_conv = convolve(self, x)
            rows = numel(self.kernel) - 1;
            x_fft = fft([x;zeros(rows,self.n_channels)]);
            self.conv_x = [self.conv_x;zeros(1,self.n_channels)];
            self.conv_x(self.step:self.step+rows,:) = ...
                self.conv_x(self.step:self.step+rows,:) + ...
                ifft(x_fft .* self.kernel_fft);
            x_conv = self.conv_x(self.step,:);
        end

        function x_next = zscore_step(self,x)
            self.update_mean(x);
            self.update_sigma(x);
            x_next = (x - self.mean) ./ self.sigma;
            self.step = self.step + 1;
        end

        function update_mean(self,x)
            self.previous_mean = self.mean;
            self.mean = self.mean + (x - self.mean) ./ self.step;
        end

        function update_sigma(self,x)
            self.M2 = self.M2 + (x - self.previous_mean) .* (x - self.mean);
            if self.step ==1
                self.sigma = sqrt(self.M2 ./ self.step);
            else
                self.sigma = sqrt(self.M2 ./ (self.step - 1));
            end
        end

    end

end
