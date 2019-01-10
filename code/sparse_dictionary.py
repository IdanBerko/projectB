class Thresholding2Func(Function):
    """
    standard thresholding
    Input:
    ------
    x - (batch_size, emb_dim, *)
        Last dimensions may be arbitrary
    emb - (emb_dim, num_emb)
    """

    @staticmethod
    def forward(ctx, input, dict, l1_coeff, num_bits=8, min_value=None, max_value=None):
        if input.size(1) != dict.size(0):
            raise RuntimeError('invalid argument: input.size(1) ({}) must be equal to dict.size(0) ({})'.
                               format(input.size(1), dict.size(0)))

        # save sizes for backward
        ctx.batch_size = input.size(0)
        ctx.num_latents = int(np.prod(np.array(input.size()[2:])))
        ctx.emb_dim = dict.size(0)
        ctx.num_emb = dict.size(1)
        ctx.input_type = type(input)

        # pursuit - thresholding
        x_reshaped = input.permute(0, 2, 3, 1).contiguous().view(ctx.batch_size * ctx.num_latents, ctx.emb_dim).t()
        alpha = dict.t() @ x_reshaped  # kx(nxnxb)
        alpha[torch.abs(alpha) < math.sqrt(2*l1_coeff)] = 0

        alpha = quantize(alpha, num_bits=num_bits, min_value=min_value, max_value=max_value)

        result = dict @ alpha

        return result.t().contiguous().view(ctx.batch_size, *input.size()[2:], ctx.emb_dim).permute(0, 3, 1, 2), \
               x_reshaped, alpha


    @staticmethod
    def backward(ctx, grad_output, x_reshaped_grad=None, alpha_grad=None, num_bits=None, min_value=None, max_value=None):
        grad_input = grad_emb = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output

        return grad_input, grad_emb, None, None, None, None


def persuit_thresholding2(x, emb, l1_coeff, num_bits=8, min_value=None, max_value=None):
    return Thresholding2Func().apply(x, emb, l1_coeff, num_bits, min_value, max_value)

class SparseDictionary(nn.Module):
    def __init__(self, num_embeddings, embeddings_dim, method='thresh', **kwargs):
        super(SparseDictionary, self).__init__()
        # assert method in ['thresh', 'thresh2', 'thresh3', 'thresh4', 'thresh5',
        #                   'rand-thresh', 'omp', 'batch-omp', 'fista']

        self.method = method
        self.pursuit_args = {}
        if method == 'thresh':
            self.pursuit = persuit_thresholding
            self.pursuit_args['num_atoms'] = kwargs.get('num_atoms')
        elif method == 'thresh2':
            self.pursuit = persuit_thresholding2
            self.pursuit_args['l1_coeff'] = kwargs.get('l1_coeff')
        elif method == 'thresh3':
            self.pursuit = Threshold3(kwargs.get('l1_coeff'), kwargs.get('num_bits'))
        elif method == 'thresh4':
            self.pursuit = persuit_thresholding4
            self.pursuit_args['l1_coeff'] = kwargs.get('l1_coeff')
        elif method == 'thresh5':
            self.pursuit = persuit_thresholding5
            self.pursuit_args['l1_coeff'] = kwargs.get('l1_coeff')
        elif method == 'thresh6':
            self.pursuit = persuit_thresholding6
            self.pursuit_args['l1_coeff'] = kwargs.get('l1_coeff')
        elif method == 'thresh7':
            self.pursuit = persuit_thresholding7
            self.pursuit_args['l1_coeff'] = kwargs.get('l1_coeff')
        elif method == 'thresh8':
            self.pursuit = persuit_thresholding8
            self.pursuit_args['l1_coeff'] = kwargs.get('l1_coeff')
        elif method == 'iter_thresh':
            self.pursuit = persuit_iter_thresholding
            self.pursuit_args['l1_coeff'] = kwargs.get('l1_coeff')
        elif method == 'rand-thresh':
            self.pursuit = persuit_random_thresholding
            self.pursuit_args['num_atoms'] = kwargs.get('num_atoms')
            self.pursuit_args['temp'] = kwargs.get('temp')
        elif method == 'omp':
            self.pursuit = persuit_omp
        elif method == 'batch-omp':
            self.pursuit = persuit_batch_omp
        elif method == 'fista':
            self.pursuit = persuit_fista
            self.pursuit_args['l1_coeff'] = kwargs.get('l1_coeff')
            self.pursuit_args['aprox_coeff'] = kwargs.get('aprox_coeff')
            self.pursuit_args['iterations'] = kwargs.get('ista_iter')
            self.pursuit_args['moment'] = kwargs.get('moment')

        self.register_buffer('weight', torch.rand(embeddings_dim, num_embeddings))
        nn.init.orthogonal_(self.weight)
        self.weight = F.normalize(self.weight, 2, 0)

        self.method = method

        if kwargs.pop('dict_update') == 'mod':
            self.weight = nn.Parameter(self.weight)
        self.num_embeddings = num_embeddings
        self.embeddings_dim = num_embeddings
        self.num_bits = kwargs.get('num_bits')
        self.register_buffer('running_min', torch.zeros(1))
        self.register_buffer('running_max', torch.zeros(1))
        self.momentum = kwargs.get('quantizer_momentum')
        self.alpha = None

    def forward(self, x, weight_sg=False):
        """Input:
        ---------
        x - (batch_size, emb_size, *)
        """
        if self.training:
            min_value = x.detach().view(
                x.size(0), -1).min(-1)[0].mean()
            max_value = x.detach().view(
                x.size(0), -1).max(-1)[0].mean()
            self.running_min.mul_(self.momentum).add_(
                min_value * (1 - self.momentum))
            self.running_max.mul_(self.momentum).add_(
                max_value * (1 - self.momentum))
        else:
            min_value = self.running_min
            max_value = self.running_max

        if self.method == 'iter_thresh':
            result, x_reshaped, alpha = self.pursuit(x, self.weight.detach() if weight_sg else self.weight,
                                                     self.alpha,
                                                     num_bits=self.num_bits, min_value=min_value, max_value=max_value,
                                                     **self.pursuit_args)
        else:
            result, x_reshaped, alpha = self.pursuit(x, self.weight.detach() if weight_sg else self.weight,
                                                     num_bits=self.num_bits, min_value=min_value, max_value=max_value,
                                                     **self.pursuit_args)
        self.x_reshaped = x_reshaped.data
        self.alpha = alpha.data
        return result

    def extra_repr(self):
        return 'num_atoms={}, emb_dim={}, method={}'.format(self.num_embeddings, self.weight.shape[0], self.method)

    def update_ksvd(self, minimal_occurrences=0):
        for i in np.random.permutation(self.num_embeddings):
            n_occurrences = torch.nonzero(self.alpha[i, :]).shape
            if n_occurrences and n_occurrences[0] > minimal_occurrences:
                # alpha kx(bxnxn)
                relevant_data_indices = torch.nonzero(self.alpha[i, :])
                tmp_coef_matrix = self.alpha[:, relevant_data_indices]
                tmp_coef_matrix[i, :] = 0  # the coefficients of the element we now improve are not relevant.
                tmp_coef_matrix.squeeze_(-1)
                error = self.x_reshaped[:, relevant_data_indices].squeeze(-1) - self.weight @ tmp_coef_matrix
                error.squeeze_(1)
                try:
                    better_dict_elements, singular_value, beta_vector = torch.svd(error)
                except RuntimeError:
                    logging.info('improper matrix. skipping SVD update')
                    continue
                self.weight[:, i] = better_dict_elements[:, 0]
                # self.alpha[i, relevant_data_indices] = singular_value[0] * beta_vector[:, 0]
                alpha_np = self.alpha.cpu().numpy()
                new_coefs = (singular_value[0] * beta_vector[:, 0]).cpu().numpy()
                alpha_np[i, relevant_data_indices] = np.expand_dims(new_coefs, -1)
                self.alpha = torch.from_numpy(alpha_np).cuda()
            else:
                # print('replacing an atom...')
                error_mat = torch.norm(self.x_reshaped - self.weight @ self.alpha, 2, 0)
                _, argmax = error_mat.max(0)
                new_atom = self.x_reshaped[:, argmax]
                new_atom = F.normalize(new_atom, 2, 0).abs()
                self.weight[:, i] = new_atom
                self.alpha[:, argmax] = 0