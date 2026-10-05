import random
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Gaussian Noise Augmentation
# =========================
class NoiseTransformation(object):
    def __init__(self, sigma: float):
        self.sigma = sigma

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        noise = torch.normal(
            mean=0.0,
            std=self.sigma,
            size=X.shape,
            device=X.device
        )
        return X + noise


# =========================
# Subsequence Anomaly Injection
# =========================
class SubAnomaly(object):
    def __init__(self, portion_len: float = 0.5):
        self.portion_len = portion_len

    def inject_frequency_anomaly(
        self,
        window: torch.Tensor,
        subsequence_length: int = None,
        compression_factor: int = None,
        scale_factor: torch.Tensor = None,
        trend_factor: float = None,
        shapelet_factor: bool = False,
        trend_end: bool = False,
        start_index: int = None
    ) -> torch.Tensor:

        window = window.clone()
        seq_len = window.shape[0]

        # ---- subsequence length ----
        if subsequence_length is None:
            min_len = max(1, int(seq_len * 0.1))
            max_len = max(min_len + 1, int(seq_len * 0.9))
            subsequence_length = torch.randint(min_len, max_len, (1,)).item()

        # ---- compression factor ----
        if compression_factor is None:
            compression_factor = torch.randint(2, 5, (1,)).item()

        # ---- scale factor ----
        if scale_factor is None:
            scale_factor = torch.rand(
                (window.shape[1],),
                device=window.device
            ) * 1.9 + 0.1  # range [0.1, 2.0]

        # ---- start index ----
        if start_index is None:
            start_index = torch.randint(
                0,
                max(1, seq_len - subsequence_length),
                (1,)
            ).item()

        end_index = min(start_index + subsequence_length, seq_len)

        if trend_end:
            end_index = seq_len

        subseq = window[start_index:end_index]

        # ---- frequency manipulation ----
        subseq = subseq.repeat(compression_factor, 1)
        subseq = subseq[::compression_factor]

        # ---- scaling ----
        subseq = subseq * scale_factor

        # ---- trend ----
        if trend_factor is None:
            trend_factor = torch.normal(
                mean=1.0,
                std=0.5,
                size=(1,)
            ).item()

        coef = -1 if torch.rand(1).item() < 0.5 else 1
        subseq = subseq + coef * trend_factor

        # ---- shapelet anomaly ----
        if shapelet_factor:
            base = window[start_index]
            subseq = base + torch.rand_like(base) * 0.1

        # ---- replace ----
        window[start_index:end_index] = subseq

        return window


    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        window = X.clone()

        seq_len = window.shape[0]

        min_len = max(1, int(seq_len * 0.1))
        max_len = max(min_len + 1, int(seq_len * 0.9))

        subsequence_length = torch.randint(min_len, max_len, (1,)).item()
        start_index = torch.randint(
            0,
            max(1, seq_len - subsequence_length),
            (1,)
        ).item()

        # ===== MULTIVARIATE =====
        if window.ndim > 1:
            num_features = window.shape[1]

            low = max(1, int(num_features / 10))
            high = max(low + 1, int(num_features / 2))

            num_dims = torch.randint(low, high, (1,)).item()

            anomaly_types = []

            for _ in range(num_dims):
                i = torch.randint(0, num_features, (1,)).item()

                temp = window[:, i].unsqueeze(1)

                anomaly_types.append(
                    self.inject_frequency_anomaly(
                        temp,
                        scale_factor=torch.ones(1, device=window.device),
                        trend_factor=0,
                        subsequence_length=subsequence_length,
                        start_index=start_index
                    )
                )

                anomaly_types.append(
                    self.inject_frequency_anomaly(
                        temp,
                        compression_factor=1,
                        scale_factor=torch.ones(1, device=window.device),
                        trend_end=True,
                        subsequence_length=subsequence_length,
                        start_index=start_index
                    )
                )

                anomaly_types.append(
                    self.inject_frequency_anomaly(
                        temp,
                        subsequence_length=2,
                        compression_factor=1,
                        scale_factor=torch.tensor([8.0], device=window.device),
                        trend_factor=0,
                        start_index=start_index
                    )
                )

                anomaly_types.append(
                    self.inject_frequency_anomaly(
                        temp,
                        subsequence_length=4,
                        compression_factor=1,
                        scale_factor=torch.tensor([3.0], device=window.device),
                        trend_factor=0,
                        start_index=start_index
                    )
                )

                anomaly_types.append(
                    self.inject_frequency_anomaly(
                        temp,
                        compression_factor=1,
                        scale_factor=torch.ones(1, device=window.device),
                        trend_factor=0,
                        shapelet_factor=True,
                        subsequence_length=subsequence_length,
                        start_index=start_index
                    )
                )

            return random.choice(anomaly_types)

        # ===== UNIVARIATE =====
        else:
            temp = window.unsqueeze(1)

            anomalies = [
                self.inject_frequency_anomaly(
                    temp,
                    scale_factor=torch.ones(1, device=window.device),
                    trend_factor=0,
                    subsequence_length=subsequence_length,
                    start_index=start_index
                ),
                self.inject_frequency_anomaly(
                    temp,
                    compression_factor=1,
                    scale_factor=torch.ones(1, device=window.device),
                    trend_end=True,
                    subsequence_length=subsequence_length,
                    start_index=start_index
                ),
                self.inject_frequency_anomaly(
                    temp,
                    subsequence_length=3,
                    compression_factor=1,
                    scale_factor=torch.tensor([8.0], device=window.device),
                    trend_factor=0,
                    start_index=start_index
                ),
                self.inject_frequency_anomaly(
                    temp,
                    subsequence_length=5,
                    compression_factor=1,
                    scale_factor=torch.tensor([3.0], device=window.device),
                    trend_factor=0,
                    start_index=start_index
                ),
                self.inject_frequency_anomaly(
                    temp,
                    compression_factor=1,
                    scale_factor=torch.ones(1, device=window.device),
                    trend_factor=0,
                    shapelet_factor=True,
                    subsequence_length=subsequence_length,
                    start_index=start_index
                )
            ]

            return random.choice(anomalies).squeeze()


# =========================
# Electricity Theft / Malicious Consumption Attack Injection (NOVELTY)
# =========================
class ElectricityTheftAttack(object):
    """
    Injeksi anomali sintetis berbasis ENAM fungsi serangan pencurian listrik
    (electricity theft cyberattack) f1-f6 yang didefinisikan oleh:

        Jokar, P., Arianpoo, N., & Leung, V. C. M. (2016). Electricity theft
        detection in AMI using customers' consumption patterns. IEEE
        Transactions on Smart Grid, 7(1), 216-226.

    dan direplikasi/dipakai sebagai skema serangan acuan pada:

        Nabil, M., Ismail, M., Mahmoud, M., Shahin, M., Qaraqe, K., &
        Serpedin, E. (2018). Deep recurrent electricity theft detection in
        AMI networks with random tuning of hyper-parameters. arXiv:1809.01774.

    NOVELTY pada penelitian ini: SubAnomaly (di atas) menyuntikkan anomali
    GENERIK yang tidak punya makna fisik khusus (manipulasi frekuensi/
    shapelet/trend pada subsequence acak). Kelas ElectricityTheftAttack ini
    menggantinya dengan anomali yang merepresentasikan PERILAKU PENCURIAN
    LISTRIK NYATA pada smart meter: pelanggan nakal memanipulasi pembacaan
    konsumsi asli Ec(d, t) menjadi pembacaan yang dilaporkan Rc(d, t) =
    f(Ec(d, t)) untuk menurunkan tagihan listriknya. Dengan memakai anomali
    sintetis yang merepresentasikan pola dunia nyata ini saat tahap pretext
    (kontrastif) dan klasifikasi CARLA, representasi yang dipelajari model
    diharapkan lebih relevan untuk domain load profile listrik (PLN)
    dibandingkan anomali generik.

    Keenam fungsi serangan (Ec = konsumsi asli, Rc = konsumsi yang dilaporkan
    setelah serangan, T = panjang window/jumlah periode dalam satu hari):

        f1  Flat Reduction Attack       Rc(t) = alpha * Ec(t)
        f2  Dynamic Reduction Attack    Rc(t) = beta(t) * Ec(t)
        f3  Selective Time Filtering /
            Bypass Attack               Rc(t) = 0 untuk t di [ti, tf],
                                         Rc(t) = Ec(t) di luar interval itu
        f4  Average Consumption Attack  Rc(t) = E[Ec]  (rata-rata harian)
        f5  Dynamic Average Attack      Rc(t) = beta(t) * E[Ec]
        f6  Load Reordering Attack      Rc(t) = Ec(T - t + 1) (balik urutan
                                         waktu, beban tinggi pindah ke jam
                                         tarif rendah)

    Catatan KOMPATIBILITAS: kelas ini murni TAMBAHAN baru dan TIDAK mengubah
    SubAnomaly maupun perilaku default-nya sedikit pun. Kelas ini hanya aktif
    jika dipilih lewat anomaly_kwargs['type'] == 'theft' (murni) atau
    'theft+subseq' (campuran, lihat MixedAnomaly di bawah) pada file
    konfigurasi (lihat inject_sub_anomaly() di utils/common_config.py). Selama
    anomaly_kwargs['type'] tidak diisi (kasus SMD/MSL/SMAP/LEAD dan konfigurasi
    PLN versi sebelumnya), seluruh pipeline tetap memakai SubAnomaly seperti
    semula -> tidak ada perubahan perilaku untuk eksperimen/checkpoint lama.

    window yang diterima/dikembalikan berbentuk (T, F): T = panjang window
    (time steps dalam satu window/hari), F = jumlah fitur/kolom. Setiap
    fungsi serangan diterapkan PER KOLOM (per fitur) pada sejumlah kolom yang
    dipilih acak, konsisten dengan cara SubAnomaly menangani kasus
    multivariat di atas.
    """

    ATTACK_CODES = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']

    def __init__(self, portion_len: float = 0.5, alpha_range=(0.1, 0.8),
                 beta_range=(0.1, 0.8)):
        self.portion_len = portion_len  # disimpan agar antarmuka selaras dgn SubAnomaly
        self.alpha_range = alpha_range
        self.beta_range = beta_range

    # ---- f1: Flat Reduction Attack: Rc(t) = alpha * Ec(t) ----
    def _f1_flat_reduction(self, col: torch.Tensor) -> torch.Tensor:
        alpha = torch.empty(1, device=col.device).uniform_(*self.alpha_range).item()
        return col * alpha

    # ---- f2: Dynamic Reduction Attack: Rc(t) = beta(t) * Ec(t) ----
    def _f2_dynamic_reduction(self, col: torch.Tensor) -> torch.Tensor:
        beta_t = self._random_beta_curve(col.shape[0], col.device)
        return col * beta_t

    # ---- f3: Selective Time Filtering / Bypass Attack ----
    def _f3_selective_bypass(self, col: torch.Tensor) -> torch.Tensor:
        col = col.clone()
        T = col.shape[0]
        min_len = max(1, int(T * 0.1))
        max_len = max(min_len + 1, int(T * 0.9))
        dur = torch.randint(min_len, max_len, (1,)).item()
        ti = torch.randint(0, max(1, T - dur), (1,)).item()
        tf = min(ti + dur, T)
        col[ti:tf] = 0.0
        return col

    # ---- f4: Average Consumption Attack: Rc(t) = E[Ec] ----
    def _f4_average(self, col: torch.Tensor) -> torch.Tensor:
        mean_val = col.mean()
        return torch.full_like(col, mean_val.item())

    # ---- f5: Dynamic Average Attack: Rc(t) = beta(t) * E[Ec] ----
    def _f5_dynamic_average(self, col: torch.Tensor) -> torch.Tensor:
        mean_val = col.mean()
        beta_t = self._random_beta_curve(col.shape[0], col.device)
        return beta_t * mean_val

    # ---- f6: Load Reordering Attack: Rc(t) = Ec(T - t + 1) ----
    def _f6_load_reorder(self, col: torch.Tensor) -> torch.Tensor:
        return torch.flip(col, dims=[0])

    def _random_beta_curve(self, length: int, device) -> torch.Tensor:
        """
        Bangkitkan beta(t) yang berubah halus terhadap waktu (bukan white
        noise per-step), sesuai gagasan f2/f5 di paper: faktor reduksi yang
        "berubah dari waktu ke waktu". Caranya: sampling beberapa control
        point acak dalam alpha_range/beta_range, lalu interpolasi linear
        sepanjang window.
        """
        n_ctrl = max(2, length // 10)
        ctrl = torch.empty(n_ctrl, device=device).uniform_(*self.beta_range)
        ctrl = ctrl.view(1, 1, n_ctrl)
        curve = torch.nn.functional.interpolate(
            ctrl, size=length, mode='linear', align_corners=True
        ).view(-1)
        return curve

    def _apply_random_attack(self, col: torch.Tensor) -> torch.Tensor:
        fn = random.choice([
            self._f1_flat_reduction,
            self._f2_dynamic_reduction,
            self._f3_selective_bypass,
            self._f4_average,
            self._f5_dynamic_average,
            self._f6_load_reorder,
        ])
        return fn(col)

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        window = X.clone()

        # ===== MULTIVARIATE =====
        if window.ndim > 1:
            num_features = window.shape[1]

            low = max(1, int(num_features / 10))
            high = max(low + 1, int(num_features / 2))
            num_dims = torch.randint(low, high, (1,)).item()

            cols_to_attack = torch.randperm(num_features)[:num_dims]
            for i in cols_to_attack.tolist():
                window[:, i] = self._apply_random_attack(window[:, i])

            return window

        # ===== UNIVARIATE =====
        else:
            return self._apply_random_attack(window)


# =========================
# Mixed Anomaly Injection (NOVELTY): theft+subseq
# =========================
class MixedAnomaly(object):
    """
    Gabungan SubAnomaly (anomali generik bawaan CARLA) + ElectricityTheftAttack
    (NOVELTY f1-f6) dalam SATU mekanisme injeksi. Dipilih lewat
    anomaly_kwargs['type'] == 'theft+subseq' (lihat inject_sub_anomaly() di
    utils/common_config.py).

    Tujuannya: bukan memilih salah satu (subseq ATAU theft), tapi mencampur
    keduanya supaya model pretext/klasifikasi belajar membedakan window normal
    dari DUA distribusi anomali sekaligus - anomali generik (frekuensi/
    shapelet/trend) dan anomali yang merepresentasikan pola pencurian listrik
    nyata (f1-f6). Tiap kali dipanggil (yaitu tiap window), salah satu dari
    dua mekanisme dipilih secara acak sesuai `theft_ratio` (default 0.5 ->
    campuran seimbang 50/50); window lain dalam dataset yang sama jadi bisa
    mendapat anomali subseq atau theft secara bergantian.

    Catatan KOMPATIBILITAS: kelas ini murni TAMBAHAN baru, tidak mengubah
    SubAnomaly maupun ElectricityTheftAttack sedikit pun, dan hanya aktif
    kalau anomaly_kwargs['type'] secara eksplisit diisi 'theft+subseq'. Semua
    konfigurasi lama ('subseq'/tidak diisi, atau 'theft') tetap berjalan
    seperti sebelumnya.
    """

    def __init__(self, portion_len: float = 0.5, theft_ratio: float = 0.5,
                 alpha_range=(0.1, 0.8), beta_range=(0.1, 0.8)):
        self.portion_len = portion_len  # disimpan agar antarmuka selaras dgn SubAnomaly/ElectricityTheftAttack
        self.theft_ratio = theft_ratio
        self._subseq = SubAnomaly(portion_len)
        self._theft = ElectricityTheftAttack(portion_len, alpha_range=alpha_range, beta_range=beta_range)

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        if random.random() < self.theft_ratio:
            return self._theft(X)
        return self._subseq(X)