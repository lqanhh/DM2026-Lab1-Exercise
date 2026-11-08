# Your report

*(Replace this with your own report: 3-5 short paragraphs. For each major step you actually ran, loading the data, filtering, pattern mining, dimension reduction, whatever applies, say what you did, reference the specific `result_id` and/or saved plot, and explain what it shows. Delete this instruction text before submitting.)*

I first loaded the dataset with categories *alt.atheism, soc.religion.christian, comp.graphics, and sci.med*. It loaded successfully with 2,257 documents, with 480, 599, 584 and 594 documents **respectively** for the four categories listed above (ID **load_dataset_1**). The dataset did not contain any missing values nor duplicates. The plot below shows basic statistics of each category.
!["Boxplots of categories"](..\plots\Luong_Quynh_Anh_agentic-pipeline_plot_describe_data_5.png)

I also tried subsample the data with a sample size of 20 to ensure that the tool works. But I did not use this subsample for the next tool runs (ID **sample_data_6**).

I then built the document-term matrix *(DTM)* (ID **dtm_7**). It contained all documents available in the dataset, with nearly 81 million elements and a sparsity of **99.547%**. After that, I checked the term frequency statistics (ID **term_freq_8**) and found the top 20 most frequent terms in the dataset. Most of them were actually stop words, so we need to proceed to the next filtering and pattern mining steps. A document-term matrix slice of 20 documents and 20 terms was also generated with **dtm_heatmap_tool** (ID **dtm_heatmap_9**) *(see below)*.
!["Document-term matrix heatmap"](..\plots\Luong_Quynh_Anh_agentic-pipeline_plot_dtm_heatmap_9.png)

To start the filtering steps, I first tried filtering terms by variance (ID **variance_filter_11**), with a threshold of 0.01. This ruled out 18,465 terms, so more than half the number of terms in the document-term matrix. I did the same thing with a higher threshold - 0.01 (ID **variance_filter_10**) and it removed 30,434 terms, leaving only 5,354 terms in the DTM. Pearson correlation filtering was then applied to the *comp.graphics* category using a one-vs-rest target and a threshold of 0.05 (ID **pearson_filter_12**). Out of 35,788 original terms, 3,418 were retained and 32,370 were removed. I looked at a sample of what is kept and most are highly related to the category (except for 'thanks', in my small sample). I did the same thing with *soc.religion.christian* (ID **pearson_filter_13**; kept 2,427 - removed 33,361), *alt.atheism* (ID **pearson_filter_14**; kept 2,368 - removed 33,420) and *sci.med* (ID **pearson_filter_15**; kept 2,699 - removed 33,089). Next, a feature correlation matrix was created (ID **feature_correlation_16**). We can see that words which are not English stop words such as edu and jpeg don't have a strong correlation with the other words.
!["Feature correlation matrix"](..\plots\Luong_Quynh_Anh_agentic-pipeline_plot_feature_correlation_16.png)

In addition, I tested the **visualize_result_tool** to plot out the top 20 terms. See the plot below:
!["Top 20 most frequent terms"](..\plots\Luong_Quynh_Anh_agentic-pipeline_plot_turn80.png)

Moving on to the pattern mining step, I started with mine_patterns_tool on *alt.atheism* (ID **mine_patterns_17**), then *sci.med* (ID **mine_patterns_18**), and *soc.religion.christian* (ID **mine_patterns_19**). The most frequent patterns are dominated by metadata terms (e.g., “subject,” “lines,” and “organization”), suggesting that the dataset contains substantial header information. Some topic-related terms, however, also appear frequently.

After the pattern mining, I reduced the dimensions of my document-term matrix down to three (ID **reduce_dimensions_3**) by the PCA method. The PCA projection shows substantial overlap among the four newsgroup categories, suggesting that the first two principal components do not clearly separate documents by topic.
!["2d projection"](..\plots\Luong_Quynh_Anh_agentic-pipeline_plot_reduce_dimensions_3.png) 

I then binarized the labels into a 2D matrix (ID **binarize_labels_4**). This produced a matrix of 2257 rows and 4 columns.
