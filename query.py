from collections import defaultdict

def top_k_acc(topK, classes, match): #int, np.array, arr
# slicing the arr until k
    match = match[:, :k]

    tot = match.shape[0] ##dim of the np array
    cor = 0
    for i, label in enumerate(classes):
        #compute the accuracy 
        cor+= np.any(label == match[i, :]).item()
    acc = cor/tot
    return acc

def query(distance_arr, path):
    for k in [1, 3, 10]:
        topk_acc = topk_accuracy(query_classes, gallery_matches, k)
        print('--> Top-{:d} Accuracy: {:.3f}'.format(k, topk_acc))
    k_match = defaultdict(list)
    viz_indices = indices[29]
    top_1 = viz_indices[0]

    img_match = cv2.imread(gallery_paths[top_1])
    img_match = cv2.cvtColor(img_match, cv2.COLOR_BGR2RGB)

    plt.imshow(img_match)
    return k_match

