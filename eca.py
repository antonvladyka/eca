import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange


class ECA(nn.Module):
    """ECA approach implemented in pytorch
    Requires torch-based emulator to predict X -> Y
    Example:
    # test_x_t and test_y_t are tensors (test X and Y data, respectively)
    # convert numpy array to torch.tensors via e.g.
    test_x_t, test_y_t = map(lambda x: torch.tensor(x, dtype=torch.float32), [test_x, test_y])
    eca = ECA(model) #  initialize with a trained predictor (for pytorch the model itself is a predictor,
        i.e., Y = model(X)  <=> Y = model.forward(X))
    options = {
        'tol'   : 1E-4,
        'epochs': 10000,
        'lr'    : 1E-3,
        'batch_size': 200,
        'seed': 123,
    }
    v, y_var, x_var = eca.fit(test_x_t, test_y_t, n_comp=3, options=options)
    # check orthonormality
    # must be unity matrix
    print(v @ v.T)
    """
    # default options
    DEFAULT_LR = 1E-3
    DEFAULT_TOL = 1E-4
    DEFAULT_EPOCHS = 10000
    DEFAULT_BETAS = (0.9, 0.999)
    DEFAULT_BATCH_SIZE = 200
    DEFAULT_LR_INV = 5E-2
    DEFAULT_TOL_INV = 1E-4
    DEFAULT_EPOCHS_INV = 1000

    def __init__(self, predict: nn.Module):
        """
        Parameters
        -----------
        predict: nn.Module or func X -> Y
            Predictor, torch-based
        """
        super().__init__()
        self.predict = predict
        self.V = None  # calculated components
        self._v = None  # currently evaluated component
        self.u = None
        self.y_var: list = []  # covered variance for Y
        self.x_var: list = []  # covered variance for X
        self.training_losses: list = []
        self.seed = None

    def set_seed(self, seed: int) -> None:
        """ Set seed for torch.manual_seed
        Affects initial guess of the vector (via xavier_init_) and batches while training """
        self.seed = seed

    @property
    def n_calc(self) -> int:
        """Number of already calculated components"""
        return self.V.size()[0] if self.V is not None else None

    def _init(self, n: int, start: int = 0) -> None:
        """
        Initializes the vectors as well as resets them after restart of the fit
        """
        self.n = n  # dimensionality of input data
        if self.V is None:
            self.V = torch.empty(0, self.n, dtype=torch.float32)  # already calculated EC components
        self._v = torch.empty(1, self.n, dtype=torch.float32, requires_grad=True)  # currently evaluated component
        self.u = torch.empty(1, self.n, dtype=torch.float32)  # to keep current vector
        assert start >= 0, 'Starting component number must be nonnegative'
        self.V = self.V[:start, ]
        self.x_var = self.x_var[:start]
        self.y_var = self.y_var[:start]
        nn.init.xavier_normal_(self._v)

    @staticmethod
    def r2loss(predicted: torch.Tensor, known: torch.Tensor) -> torch.Tensor:
        """Calculated generalized missing variance between known and predicted values.
        Covered variance = 1 - missing variance"""
        known_mean = torch.outer(torch.ones(known.shape[0]), known.mean(axis=0))
        return torch.trace((known - predicted).T @ (known - predicted)) / torch.trace(
            (known - known_mean).T @ (known - known_mean))

    @property
    def v(self) -> torch.Tensor:
        """Returns currently evaluated normalized vector"""
        return self._v / torch.linalg.vector_norm(self._v)

    @property
    def vectors(self) -> torch.Tensor:
        return self.V
    
    def __add(self) -> None:
        """After successful training, found component added to the calculated ones, and reset"""
        tmp = self.v.detach()
        self.V = torch.vstack((self.V, tmp))
        # reset the weights
        nn.init.xavier_normal_(self._v)
        # remove the projection on the known components
        with torch.no_grad():
            vx = self._v
            self._v -= self.project(vx)

    def transform(self, x: torch.Tensor, n_comp: int = None) -> torch.Tensor:
        """Calculates t-scores (i.e. projection values of X onto ECA basis space"""
        if n_comp is None:
            return x @ self.V.T
        return x @ self.V[:n_comp, ].T

    def expand(self, t: torch.Tensor, n_comp: int = None) -> torch.Tensor:
        """Expands the t-scores to X space: t -> x'
        Parameters
        -------------
            t: torch.tensor
                projections in the v-space
            n_comp: int
                Calculates only $comp components of basis space
        """
        if n_comp is None:
            return t @ self.V
        return t @ self.V[:n_comp, :]

    def project(self, x: torch.Tensor, n_comp: int = None) -> torch.Tensor:
        """Projects x onto ECA basis space"""
        return self.expand(self.transform(x, n_comp), n_comp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Used in the fit only. Projects X onto V space and predicts Y"""
        v = self.v
        proj = (x @ v.T) @ v
        if self.n_calc > 0:
            #  if some components V already calculated, projection = sum of projections on V and v
            return self.predict(self.project(x) + proj)
        else:
            return self.predict(proj)

    def fit(self, data_x: torch.Tensor, data_y: torch.Tensor, n_comp: int = 3,
            options: dict = None, verbose: bool = False, keep: int = False) -> (torch.Tensor, list, list):
        """
        Parameters
        -------------
        data_x: torch.Tensor
        data_y: torch.Tensor
        n_comp: int
            number of EC components to calculate
        options: dict
            'lr': float, default 0.001
                initial learning rate
            'betas': tuple(float, float), default (0.9, 0.999)
                beta parameters for Adam optimizer, see torch.optim.adam
            'epochs': int, default 10000
                Maximum number of epochs for training
            'batch_size': int, default 200
                Batch size for training
            'seed': int, default None
                Seed for random number generator
        verbose: bool, default False
            If True, uses tqdm.trange to indicate progress of fit
        keep: int or bool
            Allow to continue the fit starting from specific component
            if False or 0: restarts fit from the beginning
            if True: starts from already calculated components
            if N, int: start from given number of components
        Returns
        ---------
        """
        n = data_x.shape[1]
        options = options or {}
        seed = options.get('seed') or self.seed
        if isinstance(seed, int):
            torch.manual_seed(seed)
        start = 0
        if keep:
            n_comp_calculated = self.n_calc
            if isinstance(keep, bool):
                start = n_comp_calculated if keep else 0
            elif isinstance(keep, int):
                if keep > n_comp_calculated:
                    print(f'Warning: Starting from {n_comp_calculated} components instead of {keep}')
                    start = n_comp_calculated
                elif keep < 0:
                    start = n_comp_calculated + keep
                    if start < 0:
                        start = 0
                else:
                    start = keep
            else:
                raise TypeError('continue_ must be integer number or True/False')
        self._init(n, start)  # initialize V and v tensors
        lr = options.get('lr') or self.DEFAULT_LR
        betas = options.get('betas') or self.DEFAULT_BETAS
        epochs = options.get('epochs') or self.DEFAULT_EPOCHS
        batch_size = options.get('batch_size') or self.DEFAULT_BATCH_SIZE
        tol = options.get('tol') or self.DEFAULT_TOL
        optimizer = optim.Adam([self._v], lr=lr, betas=betas)
        loss_fn = self.r2loss
        train_dataset = TensorDataset(data_x, data_y)
        for comp in range(start, n_comp):
            print(f'Start component #{comp + 1}, start={start}')
            if verbose:
                epochs_ = trange(epochs)
            else:
                epochs_ = range(epochs)
            training_loss = []
            for epoch in epochs_:
                train_loader = DataLoader(train_dataset, batch_size)
                batch_loss = []
                for x_batch, y_batch in train_loader:
                    # Generate predictions
                    pred = self.forward(x_batch)
                    loss = loss_fn(pred, y_batch)
                    optimizer.zero_grad()
                    batch_loss.append(loss.item())
                    loss.backward()
                    optimizer.step()
                    if comp > 0:
                        # to ensure search of the orthogonal components, remove from the gradient its projection
                        # on the known components
                        with torch.no_grad():
                            v = self._v
                            self._v.data -= self.project(v)

                training_loss.append(sum(batch_loss) / len(batch_loss))
                if len(training_loss) > 50:
                    # take mean of the last 5 training losses
                    dl = sum(training_loss[-6:-1]) / 5 - training_loss[-1]
                    if dl < tol:
                        print(f'Accuracy change level of {dl:2.4g} has been reached after epoch #{epoch+1}')
                        if dl < 0:
                            self._v.data = self.u.data  # returns previous vector
                        break
                self.u.data = self._v.data  # keep track of the current vector
            else:
                print(f'Max number of epochs ({epoch+1}) has been reached. Accuracy change = {dl:2.4g}')
            self.training_losses.append(training_loss)
            self.__add()  # save found component and reinitialize v
            pred_y = self.predict(self.project(data_x))
            y_loss = loss_fn(pred_y, data_y).item()
            self.y_var.append(1 - y_loss)
            print(f'Covered variance for component {comp+1}:', 1-y_loss)
            x_loss = loss_fn(self.project(data_x), data_x).item()
            self.x_var.append(1 - x_loss)
        return self.V, self.y_var, self.x_var

    def _inverse(self, y: torch.Tensor, n_comp: int = None, options: dict = None) -> (torch.Tensor, torch.Tensor):
        """This method returns an approximation of the point in t-space such that the prediction from this point
        is closest to the given y.
        Note: potentially, this can be done for all Y values simultaneously but in practice convergence is bad.
        Parameters
        -----------
        y: torch.tensor
            given Y vector(-s) to calculate inverse
        n_comp: int
            if set, uses n_comp to calculate inverse transformations
        options: dict
            'optimizer': str
                    optimizer for inverse transform, 'adam' (default) or 'sgd'
            'tol_inv' or 'tol': float, default 1E-4
                tolerance level to reach upon approximation
            'epochs_inv' or 'epochs': int, default 1000
                number of epochs to run
            'lr_inv' or 'lr' : float, default 0.05
                learning rate
        """
        options = options or {}
        lr = options.get('lr_inv') or options.get('lr') or self.DEFAULT_LR_INV
        epochs = options.get('epochs_inv') or options.get('epochs') or self.DEFAULT_EPOCHS_INV
        betas = options.get('betas') or self.DEFAULT_BETAS
        tol = options.get('tol_inv') or options.get('tol') or self.DEFAULT_TOL_INV
        if n_comp is None:
            n_comp = self.n_calc
        assert 0 < n_comp <= self.n_calc, 'n_comp must be positive and not greater than number of calculated ' \
                                          'components '
        t = torch.empty(y.size()[0], n_comp, dtype=torch.float32, requires_grad=True)
        optimizer = optim.Adam([t], lr=lr, betas=betas)
        nn.init.xavier_normal_(t)
        err = None
        loss_fn = nn.MSELoss()
        training_loss = []
        for epoch in range(epochs):
            proj = self.expand(t, n_comp)  # <- t @ V[:n_comp]
            pred = self.predict(proj)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            err = loss.item()
            training_loss.append(err)
            if epoch > 100:
                dl = sum(training_loss[-6:-1]) / 5 - err
                if 0 < dl < tol:
                    break
        return t.detach(), err

    def inverse(self, y: torch.Tensor, options: dict = None, n_comp: int = None,
                verbose: bool = False, return_error: bool = False) -> (torch.Tensor, ):
        """This method returns an approximation of the point in t-space such that the prediction from this point
            is closest to the given y.
            Parameters
            -----------
            y: torch.tensor
                given Y vector(-s) to calculate inverse
            n_comp: int
                if set, uses n_comp to calculate inverse transformations
            options: dict
                tol_inv or tol: float, default 1E-4
                    tolerance level to reach upon approximation
                epochs_inv or epochs: int, default 1000
                    number of epochs to run
                lr_inv or lr: float, default 0.05
                    learning rate for optimizer
            return_error: bool, default False
                return MSE errors
            verbose: bool, default False
                to output progressbar, uses tqdm.trange
        """
        if y.ndim == 1:
            y = y[None, :]
        n = y.shape[0]
        if n_comp:
            assert 0 < n_comp <= self.n_calc, 'n_comp must be positive and not greater than' \
                                              'number of calculated components'
            t_scores = torch.zeros((n, n_comp))
        else:
            t_scores = torch.zeros((n, self.n_calc))
        err = torch.zeros(n)
        rng = trange(n) if verbose else range(n)
        for idx in rng:
            t_scores[idx, :], err[idx] = self._inverse(y[[idx]], n_comp=n_comp, options=options)
        if return_error:
            return t_scores, err
        return t_scores

    def reconstruct(self, y: torch.Tensor, options: dict = None, n_comp: int = None,
                    verbose: bool = False, return_error: bool = False) -> (torch.Tensor, ):
        """
        See help(ECA.inverse) for parameters
        """
        if return_error:
            t_scores, err = self.inverse(y, options=options, n_comp=n_comp, verbose=verbose, return_error=True)
            return self.expand(t_scores, n_comp=n_comp), err
        else:
            t_scores = self.inverse(y, options=options, n_comp=n_comp, verbose=verbose, return_error=False)
            return self.expand(t_scores, n_comp=n_comp)

    def test(self, x: torch.Tensor, y: torch.Tensor, n_comp: int = None) -> float:
        """ Calculates covered variance for the given x, y, i.e. 1 - R2(predicted from projected, known)
        """
        y_predicted = self.predict(self.project(x, n_comp))
        return 1 - self.r2loss(y_predicted, y).item()
