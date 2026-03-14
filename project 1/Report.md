## Report: Spam Detection Classification Models

## Results
====================================================================================================
Table 1: Logistic Regression Results
====================================================================================================
dataset representation gd_variant  lambda  accuracy  precision   recall       f1
 enron1            bow      batch    1.00  0.947368   0.893082 0.953020 0.922078
 enron1            bow mini-batch    0.10  0.949561   0.903846 0.946309 0.924590
 enron1            bow stochastic    0.01  0.921053   0.912409 0.838926 0.874126
 enron1      bernoulli      batch    1.00  0.956140   0.921569 0.946309 0.933775
 enron1      bernoulli mini-batch    0.10  0.958333   0.933333 0.939597 0.936455
 enron1      bernoulli stochastic    0.01  0.899123   0.940171 0.738255 0.827068
 enron2            bow      batch    1.00  0.945607   0.919355 0.876923 0.897638
 enron2            bow mini-batch    0.10  0.945607   0.926230 0.869231 0.896825
 enron2            bow stochastic    0.01  0.895397   0.900000 0.692308 0.782609
 enron2      bernoulli      batch    1.00  0.924686   0.912281 0.800000 0.852459
 enron2      bernoulli mini-batch    0.10  0.922594   0.911504 0.792308 0.847737
 enron2      bernoulli stochastic    0.01  0.882845   0.940476 0.607692 0.738318
 enron4            bow      batch    1.00  0.953959   0.939904 1.000000 0.969021
 enron4            bow mini-batch    0.10  0.953959   0.939904 1.000000 0.969021
 enron4            bow stochastic    0.01  0.953959   0.939904 1.000000 0.969021
 enron4      bernoulli      batch    1.00  0.957643   0.944444 1.000000 0.971429
 enron4      bernoulli mini-batch    0.01  0.957643   0.953202 0.989770 0.971142
 enron4      bernoulli stochastic    0.01  0.959484   0.946731 1.000000 0.972637

====================================================================================================
Table 2: Naive Bayes results for different variants
====================================================================================================
dataset representation  accuracy  precision   recall       f1
 enron1            bow  0.932018   0.927536 0.859060 0.891986
 enron1      bernoulli  0.730263   0.861111 0.208054 0.335135
 enron2            bow  0.935146   0.909091 0.846154 0.876494
 enron2      bernoulli  0.776151   0.896552 0.200000 0.327044
 enron4            bow  0.968692   0.969849 0.987212 0.978454
 enron4      bernoulli  0.917127   0.896789 1.000000 0.945586

Report:
- Multinomial Naive Bayes (BOW)
    On almost all occassions, Multinomial NB has a fairly high f1 value compared to bernoulli. 
    The accuracy averages to be 94.5%, precision 93.5%, recall 89.7%, and f1 91.6%. It has a much higher detection rate for Spam as well as reduces false positives, and negatives, yet doesn't excel as much as Logistic Regression
- Bernoulli Naive Bayes (Bernoulli)
    Bernoulli had an average accuracy of 80.8%, precision 88.5%, recall 46.9%, and f1 53.6%. Surprisingly, on enron4, bernoulli performed exceedingly well, while on the other datasets such as enron1 and enron2, it had by far the worst results amongst the three. Bernoulli NB had a poor ability to actually detect spam, shown by how poor the recall was most of the time, yet it was able to still have decent numbers regarding accuracy and precision. It's a mixed result, but with its precision being qualifiable, it's possible to conclude that it's able to reduce false positives when it decides to label something as positive (spam). Sadly, there isn't much to conclude beside it being a substandard model.
- Logistic Regression (both)
    On almost all cases, except enron4, stochastic gradient descent performed worse than batch or mini-batch. It's difficult to determine whether bernoulli or BoW did better for LR due to how volatile some of the data is, so no decisive conclusion can be made for that. Overall, logistic regression did a much better job than Naive Bayes, with an average f1 score around 94.4% for 18 datasets. 
### Hyperparameter Tuning
    Logistic Regression: Tested lambda on {0.01, 0.1, 1.0, 10.0} using 70/30 train/validation split. The lambda chosen for each gradient variant (batch,mini,SGD) was dependent on the variant itself. I tried running only batch to determine the best lambda, but it leads to stochastic being stuck with regularization constant at 1, and producing a f1 score of 0. 
    
    For learning rate, models primarily used 0.01 except for SGD which used 0.001 due to non convergence upon testing several trials. All models were trained for 500 iterations.

### Analysis:
1. Did Naive Bayes or Logistic Regression perform better? Why?
    The Logistic Regression performed far better than Naive Bayes. LR had an average F1 score of 94.1% across datasets while NB combined had 86.8%. LR likely captures non-linear decision boundaries better for spam classification.

2. Which combination of algorithm and data representation yielded the best performance? Why?
    Surprisingly, Multinomial Naive Bayes with enron4 performed the best by far with a F1 score of 97.84%, beating Logistic Regression's best by .5% (which was also on enron4). I would say that this is likely an outlier statistics, as Logistic Regression tends to have the best average F1 score. 

    As for my overall answer, LR with BoW does the best with an average F1 of 91.2% compared to Bernoulli with an average F1 of 89.5%. Given the small datasets, it's not a statstically significant difference to conclude decisive differences between the two. Personally, I do think BoW offers a far richer information plane of features than Bernoulli since counts indicate both presence and quantity, while Bernoulli can only determine a binary plane. Logistic Regression as a whole yields the best performance because the model offers a much richer learning model than Naive Bayes, allowing it to learn without hard constraints like "independence assumptions." It can model feature interactions and directly optimize classification loss.

3. Did Multinomial Naive Bayes perform better than Logistic Regression on the Bag
of Words representation? Explain.

    No, Logistic Regression outperformed Multinomial NB on BoW across all BoW datasets (avg F1 93.6% vs 91.2%). Logistic Regression was far better than Multinomial NB on recall, showcasing that not only is it more precise, it is also far better at catching actual spam. 

4. Did Bernoulli Naive Bayes perform better than Logistic Regression on the Bernoulli
representation? Explain.

    Logistic Regression significantly outperformed Bernoulli NB (avg F1 93.2% vs 53.3%). Bernoulli NB struggled on recall, reducing its F1 score drastically. It assumes feature independence, which doesn't hold well for spam detection. Bernoulli NB had results worse than all variants of LR on each dataset, sometimes around a F1 score of 30%.

5. Which variant of gradient descent was better in terms of speed and/or accuracy
when learning logistic regression classifiers.

    Batch and mini-batch GD achieved similar in accuracy with both being ~94% F1. Stochastic did much worse relative to the both of them at 85% F1. SGD oftentimes either didn't converge, got bounded by too high of a regularization constant, or converged to worse local minima. It was a difficult balance between too slow of a learning rate (bad convergence) or too high (never converged). 



    