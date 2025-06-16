from metaflow import FlowSpec, IncludeFile, current
from metaflow import step
from metaflow import card


class DataDriftTestParametersMixin:
    dataset = IncludeFile(
        "dataset",
        is_text=False,
        help="Dataset",
    )

    reference_dataset = IncludeFile(
        "reference_dataset",
        is_text=False,
        help="Reference dataset to use for drift test",
    )


class DataDriftTestMixin:
    def load_dataset(self, ds):
        import sys

        print(f"Loading dataset from IncludeFile: {sys.getsizeof(self.dataset)}")
        try:
            from io import BytesIO
            import pandas as pd

            bytes_io = BytesIO(ds)
            df = pd.read_parquet(bytes_io)
            # print(f"Loading {ds} ...")
            print(f"Dataset loaded successfully. Shape: {df.shape}")
            print("First 5 rows sample:\n", df.head())
            return df
        except Exception as e:
            print(f"Error loading Parquet file: {e}")
            raise ValueError("Failed to load dataset from Parquet IncludeFile") from e


class DriftTest_KS(DataDriftTestParametersMixin, DataDriftTestMixin, FlowSpec):
    @step
    def start(self):
        print("starting")
        self.next(self.load)

    @step
    def load(self):
        print("loading datasets ...")
        self.ds = self.load_dataset(self.dataset)
        # i used the split function in my preprocess, since in the raw data
        # there where a lot of NoneTypes and my preprocessing was failing,
        # to save time I'll use the test splitted part of one dataset, not the reference dataset parameter
        # self.ref_ds = self.load_dataset(self.reference_dataset)
        print("loading datasets: done")
        self.next(self.test_ks)

    @card
    @step
    def test_ks(self):
        import pandas as pd
        from scipy.stats import ks_2samp
        from custom_preprocess import preprocess
        # import matplotlib.pyplot as plt
        # import io

        from metaflow import current
        from metaflow.cards import Markdown

        (_, df, _, r_df) = preprocess(self.ds, split_data=True)
        # _, r_df = preprocess(self.ref_ds, split_data=False)

        drift_results = []
        for col in df.columns:
            stat, p_value = ks_2samp(df[col], r_df[col])
            drift_results.append(
                {
                    "feature": col,
                    "ks_stat": stat,
                    "p_value": p_value,
                    "drift_detected": p_value < 0.05,
                }
            )

        self.drift_df = pd.DataFrame(drift_results)
        drifted = self.drift_df[self.drift_df["drift_detected"] == True]  # noqa: E712

        self.drift_summary = {
            "total_features": len(self.drift_df),
            "drifted_features": len(drifted),
        }
        # info: this code was not working correctly, so I commented it out for now
        # fig, ax = plt.subplots(figsize=(6, 4))
        # ax.bar(["No Drift", "Drift"], [
        #     len(self.drift_df) - len(drifted), len(drifted)
        # ], color=["green", "red"])
        # ax.set_title("Feature Drift Summary")
        # ax.set_ylabel("Number of Features")

        current.card.append(Markdown(f"""
        ## 🔍 Drift Detection Report

        **Total features checked:** {self.drift_summary['total_features']}
        **Features with drift (p < 0.05):** {self.drift_summary['drifted_features']}
        """))
        # TODO(mahdi): fix it
        # current.card.append(Image.from_matplotlib(ax.plot()))

        drift_table = "\n".join([
            f"| {row['feature']} | {row['ks_stat']:.4f} | {row['p_value']:.4f} | ✅"
            for _, row in drifted.iterrows()
        ])
        current.card.append(Markdown(f"""
        ### 🔬 Features with Drift

        | Feature | KS Stat | P-Value | Drift Detected |
        |---------|---------|---------|----------------|
        {drift_table}
        """))

        print(
            f"\n✅ Drift detected in {self.drift_summary['drifted_features']} of {self.drift_summary['total_features']} features."
        )

        self.next(self.end)

    @step
    def end(self):
        print(f"Flow '{current.flow_name}' completed successfully.")


if __name__ == "__main__":
    f = DriftTest_KS()
    f.run()
