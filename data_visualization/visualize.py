import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

data = pd.read_csv("../../4989---amazon-reviews/sam_junk/csv8.csv", low_memory=False)

nona = data.dropna(axis=0, how="any")

overall = nona.loc[:, "overall"]
vote = nona.loc[:, "vote"]
# review_text = nona.loc[:, "reviewText"]
# summary = nona.loc[:, "summary"]
# verified = data.loc[:, "verified"]
# image = nona.loc[:, "image"]

review_num_adj = nona.loc[:, "review_num_adj"]
review_num_verb = nona.loc[:, "review_num_verb"]
summary_num_adj = nona.loc[:, "summary_num_adj"]
review_num_len = nona.loc[:, "review_num_length"]
summary_num_len = nona.loc[:, "summary_num_length"]

# plt.scatter(review_num_len, vote)
# plt.xlabel("Length of Review")
# plt.ylabel("Number of Helpfulness Votes")
# plt.savefig("charts/reviewlen_votes.png")
#
# plt.scatter(summary_num_len, vote)
# plt.xlabel("Length of Review Summary")
# plt.ylabel("Number of Helpfulness Votes")
# plt.savefig("charts/summarylen_votes.png")
#
# plt.scatter(review_num_adj, vote)
# plt.xlabel("Review Adjective Count")
# plt.ylabel("Number of Helpfulness Votes")
# plt.savefig("charts/reviewnumadj_votes.png")

# plt.scatter(overall, review_num_len)
# plt.xlabel("Overall Rating of Product")
# plt.ylabel("Length of Review")
# plt.savefig("charts/overall_reviewnumlen.png")

plt.scatter(overall, review_num_len)
plt.xlabel("Overall Rating of Product")
plt.ylabel("Length of Review")
plt.show()

