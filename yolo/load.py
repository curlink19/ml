import onnx
import torch
from yolo import YOLOV5m
import config


def get_dict_with_weights():
    def get_dist(graph, start, flag=False):
        res = dict()
        res[start] = 0
        q = [start]
        pos = 0
        while pos < len(q):
            for v in graph[q[pos]]:
                if not flag:
                    if "conv" in v.lower():
                        t = 1
                    else:
                        t = 0
                else:
                    if "conv" in v.lower():
                        t = 1
                    elif "mul" in v.lower():
                        t = 1
                    elif "add" in v.lower():
                        t = 1
                    else:
                        t = 0

                if v not in res or res[v] > res[q[pos]] + t:
                    res[v] = res[q[pos]] + t
                    q.append(v)
            pos += 1
        return res

    def get_bidist(model_graph, start, end):
        graph = {}
        reversed_graph = {}
        arr_conv = []
        for node in model_graph.node:
            if "tity" in node.name or "constant" in node.name.lower():
                continue
            if "onv" in node.name.lower():
                arr_conv.append(node.output[0])
            graph[node.output[0]] = list()
            reversed_graph[node.output[0]] = list()

        for node in model_graph.node:
            for v in node.input:
                if v not in graph or node.output[0] not in graph:
                    continue
                graph[v].append(node.output[0])
                reversed_graph[node.output[0]].append(v)

        return (
            (get_dist(graph, start, False), get_dist(graph, start, True)),
            (
                get_dist(reversed_graph, end, False),
                get_dist(reversed_graph, end, True),
            ),
            arr_conv,
        )

    # in onnx
    def get_names(model_a, model_b):
        graph_a = model_a.graph
        graph_b = model_b.graph
        # get_dist(graph_a)
        dist1_a, dist2_a, arr_conv_a = get_bidist(
            graph_a,
            "/backbone.0/cbl/cbl.0/Conv_output_0",
            "/neck.7/c_out/cbl/cbl.0/Conv_output_0",
        )
        dist1_b, dist2_b, arr_conv_b = get_bidist(
            graph_b,
            "/model.0/conv/Conv_output_0",
            "/model.23/cv3/conv/Conv_output_0",
        )
        print(len(arr_conv_a), len(arr_conv_b))
        res = list()
        for k_a in dist1_a[0].keys():
            if k_a not in dist2_a[0] or "conv" not in k_a.lower():
                continue

            # print(k_a, dist1_a[k_a], dist2_a[k_a])

            for k_b in dist1_b[0].keys():
                if k_b not in dist2_b[0] or "conv" not in k_b.lower():
                    continue

                if (
                    dist1_a[0][k_a] == dist1_b[0][k_b]
                    and dist2_a[0][k_a] == dist2_b[0][k_b]
                    and dist1_a[1][k_a] == dist1_b[1][k_b]
                    and dist2_a[1][k_a] == dist2_b[1][k_b]
                ):
                    res.append((k_a, k_b))
        return res, graph_a, graph_b, arr_conv_a, arr_conv_b

    a = onnx.load("pretrained/model.onnx")
    b = onnx.load("pretrained/yolov5m.onnx")
    res_list, graph_a, graph_b, arr_conv_a, arr_conv_b = get_names(a, b)
    res = dict()
    for x, y in res_list:
        if x not in res:
            res[x] = list()
        res[x].append(y)

    res["/backbone.7/cbl/cbl.0/Conv_output_0"] = [
        "/model.7/conv/Conv_output_0"
    ]
    res["/neck.3/seq/seq.1/c1/cbl/cbl.0/Conv_output_0"] = [
        "/model.17/m/m.1/cv1/conv/Conv_output_0"
    ]
    res["/neck.1/c_skipped/cbl/cbl.0/Conv_output_0"] = [
        "/model.13/cv2/conv/Conv_output_0"
    ]
    res["/backbone.8/c_skipped/cbl/cbl.0/Conv_output_0"] = [
        "/model.8/cv2/conv/Conv_output_0"
    ]
    res["/neck.3/seq/seq.1/c2/cbl/cbl.0/Conv_output_0"] = [
        "/model.17/m/m.1/cv2/conv/Conv_output_0"
    ]
    res["/neck.5/c1/cbl/cbl.0/Conv_output_0"] = [
        "/model.20/cv1/conv/Conv_output_0"
    ]
    res["/neck.1/c_out/cbl/cbl.0/Conv_output_0"] = [
        "/model.13/cv3/conv/Conv_output_0"
    ]
    res["/neck.5/seq/seq.0/c1/cbl/cbl.0/Conv_output_0"] = [
        "/model.20/m/m.0/cv1/conv/Conv_output_0"
    ]
    res["/backbone.8/c_out/cbl/cbl.0/Conv_output_0"] = [
        "/model.8/cv3/conv/Conv_output_0'"
    ]
    res["/neck.2/cbl/cbl.0/Conv_output_0"] = ["/model.14/conv/Conv_output_0"]
    res["/neck.5/seq/seq.1/c1/cbl/cbl.0/Conv_output_0"] = [
        "/model.20/m/m.1/cv1/conv/Conv_output_0"
    ]
    res["/neck.5/seq/seq.1/c2/cbl/cbl.0/Conv_output_0"] = [
        "/model.20/m/m.1/cv2/conv/Conv_output_0"
    ]
    res["/neck.7/c1/cbl/cbl.0/Conv_output_0"] = [
        "/model.23/cv1/conv/Conv_output_0"
    ]
    res["/neck.7/seq/seq.0/c1/cbl/cbl.0/Conv_output_0"] = [
        "/model.23/m/m.0/cv1/conv/Conv_output_0"
    ]

    revres = dict()
    for x, y in res.items():
        revres[y[0]] = x

    print(len(revres.keys()))

    state_dict = torch.load("pretrained/yolov5m.torchscript").state_dict()

    model = YOLOV5m(
        first_out=config.FIRST_OUT,
        nc=config.NC,
        anchors=config.ANCHORS,
        ch=(config.FIRST_OUT * 4, config.FIRST_OUT * 8, config.FIRST_OUT * 16),
        inference=False,
    )
    res_sd = model.state_dict()
    fn_pr = list()
    for k, v in state_dict.items():
        if "conv" not in k.lower():
            continue
        flag = False
        incur = None
        pos = None
        for node in graph_b.node:
            if len(node.input) > 1 and node.input[1] == k:
                flag = True
                if node.output[0] in revres:
                    incur = revres[node.output[0]]
                    pos = 1
                    # print(".".join(cur.split("/")[:-1]))
                break
            if len(node.input) > 2 and node.input[2] == k:
                flag = True
                if node.output[0] in revres:
                    incur = revres[node.output[0]]
                    pos = 2
                    # print(".".join(cur.split("/")[:-1]))
                break
        assert flag
        if not incur:
            continue
        cur = "".join(incur.split("/")[:-1])
        cur = cur.replace(".", "")
        cur = cur.replace("_", "")
        cur = cur.replace("cblcbl", "cbl")
        cur = cur.replace("seqseq", "seq")

        flag = False
        intok = None
        for tok in res_sd.keys():
            # if pos == 1 and mn != "weight":
            #    continue
            # if pos == 2 and mn != "bias":
            #    continue
            tok_form = "".join(tok.split(".")[:-1])
            tok_form = tok_form.replace("_", "")
            if tok_form == cur:
                assert not flag
                intok = tok
                flag = True
        assert flag
        if pos == 2:
            continue
        fn_pr.append((intok, v))

    print(fn_pr[0][1][0, 0, 0, 0], res_sd[fn_pr[0][0]][0, 0, 0, 0])

    for k, v in fn_pr:
        res_sd[k] = v

    print(len(fn_pr))

    return res_sd
