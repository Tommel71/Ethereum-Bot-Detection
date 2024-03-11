import pandas as pd
from sklearn.metrics import cohen_kappa_score
from tools import save_text
from tools import save_data_for_figure
from sklearn.metrics import confusion_matrix
from tools import save_table

# read in excel file
def run(prefix):
    file_path = f"{prefix}/data/wallets_to_annotate_manual.xlsx"
    df = pd.read_excel(file_path)

    # specify columns to use for calculation
    col_A = "annotatorA"
    col_B = "annotatorB"
    col_C = "annotatorC"
    col_label_A = "label_annotatorA"
    col_label_B = "label_annotatorB"
    col_A_reconciled = "A_reconciled"
    col_B_reconciled = "B_reconciled"

    # calculate cohen's kappa
    cohen_kappa = cohen_kappa_score(df[col_A], df[col_B])
    # round to two digits
    cohen_kappa = round(cohen_kappa, 2)

    percentage_agreement = (df[col_A] == df[col_B]).sum() / len(df)
    # round to 2
    percentage_agreement = round(percentage_agreement, 2)
    # contains the label names of the annotators in their own columns, the column "reconciled" contains the labels they
    # should be mapped to
    label_mapping = pd.read_csv(f"{prefix}/data_lightweight/label_mapping.csv")
    # convert each cell into a list and split on "|" except for the reconciled column
    label_mapping["annotatorA"] = label_mapping["annotatorA"].apply(lambda x: x.split("|"))
    label_mapping["annotatorB"] = label_mapping["annotatorB"].apply(lambda x: x.split("|"))

    # print result
    save_text(str(cohen_kappa), "cohens_kappa", "data", prefix)
    save_text(str(percentage_agreement), "percentage_agreement", "data", prefix)


    cohens_kappa_mapping = {
        -1: "no agreement",
        0.000000000000001: "slight agreement",
        0.2: "fair agreement",
        0.4: "moderate agreement",
        0.6: "substantial agreement",
        0.8: "almost perfect agreement",
        1: "perfect agreement",
    }

    kappa_category = ""
    for key, value in cohens_kappa_mapping.items():
        if cohen_kappa >= key:
            kappa_category = value

    save_text(kappa_category, "cohens_kappa_category", "data", prefix)

    def vote(df):
        """
        Calculate the vote of three annotators
        :param df: pandas dataframe
        :return: pandas dataframe with vote
        """

        reconciledA = df[col_label_A].apply(lambda x: label_mapping[[x in y for y in label_mapping[col_A]]]["reconciled"].values[0])
        reconciledB = df[col_label_B].apply(lambda x: label_mapping[[x in y for y in label_mapping[col_B]]]["reconciled"].values[0])
        # create empty dataframe
        data_list = []
        # iterate over all labels
        # if annotator 1 and 2 agree, the label is the label of annotator 1
        # if annotator 1 and 2 disagree, the label is the label of annotator 3
        data_detailed_list = []
        for i in range(len(df)):
            data_detailed = {"fine-grained annotator A": df[col_label_A][i],
                             "fine-grained annotator B": df[col_label_B][i]}


            if df[col_A][i] == df[col_B][i]:
                data = {"EOA": df["fromAddress"][i], "Label": df[col_A][i]}
                data_detailed.update({"fine-grained reconciled": reconciledA[i]})
                
                data_detailed.update({"Annotator C": "<not required>"})
                if df[col_label_A][i] != df[col_label_B][i]: # superfluous but there to make it explicit
                    data_detailed.update({"fine-grained reconciled": reconciledA[i]})
                
            else:
                data_detailed.update({"Annotator C":df[col_C][i]})
                
                if df[col_C][i] == "A":
                    data = {"EOA": df["fromAddress"][i], "Label": df[col_A][i]}
                    data_detailed.update({"fine-grained reconciled": reconciledA[i]})
                elif df[col_C][i] == "B":
                    data = {"EOA": df["fromAddress"][i], "Label": df[col_B][i]}
                    data_detailed.update({"fine-grained reconciled": reconciledB[i]})
                else:
                    raise Exception(f"Annotator 3 has to choose between A and B for address {df['fromAddress'][i]}")


            data_list.append(data)
            data_detailed_list.append(data_detailed)

        # count the number of disagreements
        count = (df[col_A] != df[col_B]).sum()
        
        # count the number of fine grained disagreements

        count_fine_grained = (reconciledA != reconciledB).sum()

        # create a column called "agreement" that says "bot" if the annotators agree and "human" if they disagree, "unclear" otherwise
        df["agreement"] = df.apply(lambda x: "bot" if (x[col_A] == x[col_B]) and x[col_B] == 1  else "human" if (x[col_A] == x[col_B]) and x[col_B] == 0 else "unclear", axis=1)
        counts = df["agreement"].value_counts()
        agreed_human, agreed_bot, unclear = counts["human"], counts["bot"], counts["unclear"]
        percentage = round(agreed_human / (agreed_human + agreed_bot + unclear), 2)
        save_text(str(agreed_human), "agreed_human", "data", prefix)
        save_text(str(agreed_bot), "agreed_bot", "data", prefix)
        save_text(str(unclear), "unclear", "data", prefix)
        # save count
        save_text(str(count), "annotatorA_annotatorB_n_disagreements", "data", prefix)
        save_text(str(count_fine_grained), "annotatorA_annotatorB_n_disagreements_fine_grained", "data", prefix)

        df_vote = pd.DataFrame(data_list)
        df_vote.columns=["EOA", "Bot"]
        df_vote_detailed_part = pd.DataFrame(data_detailed_list)
        df_vote_detailed = pd.concat([df_vote, df_vote_detailed_part], axis=1)

        df_vote = df_vote.set_index("EOA", drop=True)

        # reorder
        df_vote_detailed = df_vote_detailed[["EOA", "fine-grained annotator A", "fine-grained annotator B", "Annotator C",
                                             "fine-grained reconciled", "Bot"]]

        df_vote_detailed[col_A_reconciled] = reconciledA
        df_vote_detailed[col_B_reconciled] = reconciledB

        df_vote_detailed["agreement"] = df["agreement"]

        return df_vote, df_vote_detailed


    df_voted, df_vote_detailed = vote(df)
    df_vote_detailed.to_csv(f"{prefix}/data_lightweight/vote_detailed.csv", index=False)
    mask = df_voted["Bot"] == 0
    df_pretty = df_voted.copy()
    df_pretty[mask] = "Non-Bot"
    df_pretty[~mask] = "Bot"
    df_pretty.columns = ["Category"]
    counts = df_pretty.value_counts()
    n_nonbot, n_bot = counts["Non-Bot"],counts["Bot"]
    save_text(n_nonbot, "n_nonbot", "data", prefix)
    save_text(n_bot, "n_bot", "data", prefix)

    percentage_bot = round(n_bot / (n_nonbot + n_bot), 2)
    save_text(percentage_bot, "percentage_bot", "data", prefix)
    percentage_human = round(n_nonbot / (n_nonbot + n_bot), 2)
    save_text(percentage_human, "percentage_human", "data", prefix)

    df_agg_A = df_vote_detailed.groupby([col_A_reconciled]).count()
    df_agg_A = df_agg_A.sort_values(by="EOA", ascending=False)["EOA"]
    save_data_for_figure(df_agg_A, "EOAs_per_label_annotatorA", "data", prefix)

    df_agg_B = df_vote_detailed.groupby([col_B_reconciled]).count()
    df_agg_B = df_agg_B.sort_values(by="EOA", ascending=False)["EOA"]
    save_data_for_figure(df_agg_B, "EOAs_per_label_annotatorB", "data", prefix)

    # confusion matrix between a and b
    df_confusion = pd.DataFrame()
    mask = df_vote_detailed["agreement"] == "unclear"
    df_confusion["A"] = df_vote_detailed[col_A_reconciled][mask]
    df_confusion["B"] = df_vote_detailed[col_B_reconciled][mask]
    # calculate confusion
    confusion = confusion_matrix(df_confusion["A"], df_confusion["B"], labels=df_confusion["A"].unique())
    confusion_df = pd.DataFrame(confusion, index=df_confusion["A"].unique(), columns=df_confusion["A"].unique())

    save_data_for_figure(confusion_df, "confusion_annA_annB", "data", prefix)
    feature_difference_categories = df[["agreement", "max in tx in the same block", "tx per day", "n txs",  "n self transactions"]].iloc[:-1].groupby(["agreement"]).mean()

    # make ready for presentation, nice names etc
    feature_difference_categories = feature_difference_categories.rename(index={"bot": "Bot", "human": "Human", "unclear": "Unclear"})
    feature_difference_categories = feature_difference_categories.rename(
        columns={"max in tx in the same block": "Max. in-TXs per block", "tx per day": "TXs per day",
                 "n txs": "Total TXs", "n self transactions": "Self-TXs"})
    feature_difference_categories.index.name = "Agreement"
    save_table(feature_difference_categories, "feature_difference_categories", "data", prefix)



if __name__ == "__main__":
    prefix = "../.."
    run(prefix)