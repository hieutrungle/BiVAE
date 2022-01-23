import numpy as np
import scipy


class Evaluator():
    """
    include method to evaluate compression quality and error
    """

    def __init__(self, orig_in, pred_out):
        assert orig_in.shape == pred_out.shape, "[ERROR] Two inputs must have the same shape"
        self.__orig_in = orig_in
        self.__pred_out = pred_out
        self.__value_range_in = np.max(orig_in) - np.min(orig_in)
        self.max_abs_error = -1
        self.max_rel_error = -1
        self.mse = -1
        self.rmse = -1
        self.nrmse = -1
        self.psnr = -1
        self.cr = -1
        self.br = -1

    def get_input_value_range(self):
        return self.__value_range_in

    # point-wise absolute error
    def cal_max_abs_error(self) -> float:
        if self.max_abs_error == -1:
            self.max_abs_error = np.max(np.array(self.__orig_in - self.__pred_out))
        return self.max_abs_error

    def cal_abs_error(self):
        return np.array(self.__orig_in - self.__pred_out)

    def cal_total_abs_error(self) -> float:
        return np.sum(np.abs(self.cal_abs_error()))

    # point-wise relative error
    def cal_max_rel_error(self) -> float:
        if self.max_rel_error == -1:
            self.max_rel_error = self.cal_max_abs_error() / self.__value_range_in
        return self.max_rel_error

    def cal_rel_error(self):
        return self.cal_abs_error() / self.__value_range_in

    def cal_total_rel_error(self) -> float:
        return np.sum(np.abs(self.cal_rel_error()))

    # Mean square error
    def cal_mse(self) -> float:
        self.mse = np.mean(np.power(self.cal_abs_error(), 2))
        return self.mse

    def cal_rmse(self) -> float:
        if self.mse == -1:
            self.mse = self.cal_mse()
        self.rmse = np.sqrt(self.mse)
        return self.rmse

    def cal_nrmse(self) -> float:
        if self.rmse == -1:
            self.rmse = self.cal_rmse()
        self.nrmse = self.rmse / self.__value_range_in
        return self.nrmse

    # Peak signal-to-noise ratio
    def cal_psnr(self) -> float:
        if self.nrmse == -1:
            self.nrmse = self.cal_nrmse
        self.psnr = -20 * np.log10(self.cal_nrmse())
        return self.psnr

    # compression ratio
    def cal_cr(self, orig_filesize, compressed_filesize) -> float:
        self.cr = orig_filesize / compressed_filesize
        return self.cr

    # bit rate
    def cal_br(self, orig_filesize, compressed_filesize) -> float:
        if self.cr == -1:
            self.cr = self.cal_cr(orig_filesize, compressed_filesize)
        self.br = np.take(orig_filesize,0).nbytes*8 / self.cr
        return self.br

    # Pearson correlation
    def cal_correlation(self) -> float:
        return scipy.stats.pearsonr(self.__orig_in, self.__pred_out)

    # Autocorrelation
    def cal_autocorrelation(self):
        autocorrelation = np.correlate(self.__orig_in, self.__pred_out, mode="full")
        min_value = np.min(autocorrelation)
        avg_value = np.mean(autocorrelation)
        max_value = np.max(autocorrelation)
        print(f"Autocorrelation: min: {min_value} - average: {avg_value} - max: {max_value}")
        return min_value, avg_value, max_value

    def print_metrics(self):
        print(f"Mean square error (MSE): {self.cal_mse():.4f}")
        print(f"Root mean square error (RMSE): {self.cal_rmse():.4f}")
        print(f"PNSR: {self.cal_psnr():.4f}")
        print(f"Input value range: {self.__value_range_in:.4f}")
        print(f"maximum absolute error: {self.cal_max_abs_error():.3e}")
        print(f"maximum relative error: {self.cal_max_rel_error():.3e}\n")

if __name__ == "__main__":
    a = np.array([[2, 1, 3], [3, 1, 5]])
    b = np.array([[3, 6, 1], [2, 3, 9]])
    evaluator = Evaluator(a, b)

    abs_err = a - b
    value_range = np.max(a) - np.min(a)
    mse = np.mean(np.power(abs_err, 2))
    print(f"manual mse: {mse}")
    print(f"Evaluator mse: {evaluator.cal_mse()}")
    assert evaluator.cal_mse() == mse, "cal_mse is incorrect" 

    rmse = np.sqrt(mse)
    assert evaluator.cal_mse() == mse, "cal_mse is incorrect" 