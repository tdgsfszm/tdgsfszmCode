#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <thread>
#include <mutex>
#include <queue>
#include <stdio.h>
#include <sys/timeb.h>
#include <time.h>
#include <boost/type_index.hpp>
#include <tensorflow/cc/ops/image_ops.h>
#include <tensorflow/cc/ops/const_op.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/cc/ops/math_ops.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/util/command_line_flags.h>
#include <tensorflow/core/lib/io/path.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/stringpiece.h>

using namespace tensorflow;
#define BATCH_SIZE 16

Status ReadLabelsFile(const string& file_name, std::vector<std::string>* result, size_t* found_label_count) {
    std::ifstream file(file_name);
    if (!file) {
        return tensorflow::errors::NotFound("Lables file ", file_name, " not found");
    }
    result->clear();
    std::string line;
    while (std::getline(file, line)) {
        result->emplace_back(line);
    }
    *found_label_count = result->size();
    const int padding = 16;
    while (result->size() % padding) {
        result->emplace_back();
    }
    return Status::OK();
}

static Status ReadEntireFile(tensorflow::Env* env, const string& filename, Tensor* output) {
    tensorflow::uint64 filesize = 0;
    //// stores the size of filename in fileszie
    TF_RETURN_IF_ERROR(env->GetFileSize(filename, &filesize));
    tensorflow::string constents;
    constents.resize(filesize);
    std::unique_ptr<tensorflow::RandomAccessFile> file;
    TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));
    tensorflow::StringPiece data;
    TF_RETURN_IF_ERROR(file->Read(0, filesize, &data, &(constents)[0]));
    if (data.size() != filesize) {
        return tensorflow::errors::DataLoss("Truncated read of '", filename, "' expected ",
                filesize, " got ", data.size());
    }
    output->scalar<tensorflow::string>()() = tensorflow::string(data);
    return Status::OK();
}

Status LoadGraph(const string& graph_file_name, std::unique_ptr<tensorflow::Session>* session) {
    tensorflow::GraphDef graph_def;
    Status load_graph_status = ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    if (!load_graph_status.ok()) {
        return tensorflow::errors::NotFound("Failed to load graph at '", graph_file_name, "'.");
    }
    session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    Status session_create_status = (*session)->Create(graph_def);
    if (!session_create_status.ok()) {
        return session_create_status;
    }
    LOG(INFO) << "Load graph Done.";
    return Status::OK();
}

Status GetTopLabels(const std::vector<Tensor>& outputs, int how_many_labels, Tensor* indices, Tensor* scores) {
    auto root = tensorflow::Scope::NewRootScope();
    tensorflow::string output_name = "top_k";
    tensorflow::ops::TopK(root.WithOpName(output_name), outputs[0], 1);
    tensorflow::GraphDef graph;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));
    std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_RETURN_IF_ERROR(session->Create(graph));

    std::vector<Tensor> out_tensors;
    TF_RETURN_IF_ERROR(session->Run({}, {output_name + ":0", output_name + ":1"}, {}, &out_tensors));
    *scores = out_tensors[0];
    *indices = out_tensors[1];
    return Status::OK();
}

Status PrintTopLabels(const std::vector<Tensor>& outputs, const tensorflow::string& labels_file_name) {
    std::vector<tensorflow::string> labels;
    size_t label_count;
    Status read_labels_status = ReadLabelsFile(labels_file_name, &labels, &label_count);
    if (!read_labels_status.ok()) {
        return read_labels_status;
    }
    const int how_many_labels = std::min(BATCH_SIZE, static_cast<int>(label_count));
    Tensor indices;
    Tensor scores;
    TF_RETURN_IF_ERROR(GetTopLabels(outputs, how_many_labels, &indices, &scores));
    tensorflow::TTypes<float >::Flat score_flat = scores.flat<float>();
    tensorflow::TTypes<tensorflow::int32 >::Flat indices_flat = indices.flat<tensorflow::int32 >();
    for (int pos=0; pos<how_many_labels; ++pos) {
        const int label_index = indices_flat(pos);
        const float score = score_flat(pos);
        LOG(INFO) << labels[label_index] << "(" << label_index << "): " << score;
    }
    return Status::OK();
}

Status CheckTopLabel(const std::vector<Tensor>& outputs, int expected, bool* is_expected) {
    *is_expected = false;
    Tensor indices;
    Tensor scores;
    const int how_many_labels = 1;
    TF_RETURN_IF_ERROR(GetTopLabels(outputs, how_many_labels, &indices, &scores));
    tensorflow::TTypes<tensorflow::int32 >::Flat indices_flat = indices.flat<tensorflow::int32>();
    if (indices_flat(0) != expected) {
        LOG(INFO) << "Expected label #" << expected << ", but got #" << indices_flat(0);
        *is_expected = false;
    } else {
        *is_expected = true;
    }
    return Status::OK();
}

Status ReadTensorFromImageFile(const string& file_name, const int input_height,
        const int input_width, const float input_mean, const float input_std,
        std::vector<Tensor>* outTensor) {
    auto root = Scope::NewRootScope();
    string input_name = "file_reader";
    string output_name = "normalizer";
    Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
    TF_RETURN_IF_ERROR(ReadEntireFile(Env::Default(), file_name, &input));
    auto file_reader = tensorflow::ops::Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);
    std::vector<std::pair<tensorflow::string, tensorflow::Tensor>> inputs = { {"input", input} };

    const int wanter_channel = 3;
    tensorflow::Output image_reader;
    if (tensorflow::str_util::EndsWith(file_name, ".png")) {
        image_reader = tensorflow::ops::DecodePng(root.WithOpName("png_decoder"), file_reader,
                tensorflow::ops::DecodePng::Channels(wanter_channel));
    } else if (tensorflow::str_util::EndsWith(file_name, ".gif")) {
        image_reader = tensorflow::ops::Squeeze(root.WithOpName("gif_decoder"), file_reader);
    } else if (tensorflow::str_util::EndsWith(file_name, ".bmp")) {
        image_reader = tensorflow::ops::DecodeBmp(root.WithOpName("bmp_decoder"), file_reader);
    } else if (tensorflow::str_util::EndsWith(file_name, ".jpg")) {
        image_reader = tensorflow::ops::DecodeJpeg(root.WithOpName("jpeg_decoder"), file_reader,
                tensorflow::ops::DecodeJpeg::Channels(wanter_channel));
    }
    auto float_caster = tensorflow::ops::Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);
    auto dims_expander = tensorflow::ops::ExpandDims(root, float_caster, 0);
    auto resized = tensorflow::ops::ResizeBilinear(root, dims_expander,
            tensorflow::ops::Const(root.WithOpName("size"), {input_height, input_width}));
    auto transpose = tensorflow::ops::Transpose(root.WithOpName("transpose"), resized, {0, 2, 1, 3});
    tensorflow::ops::Div(root.WithOpName(output_name), tensorflow::ops::Sub(root, transpose, {input_mean}), {input_std});
    tensorflow::GraphDef graph;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));
    std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_RETURN_IF_ERROR(session->Create(graph));
    TF_RETURN_IF_ERROR(session->Run({inputs}, {output_name}, {}, outTensor));
    return Status::OK();
}

Status getMatTensor(tensorflow::Tensor& input_tensor, const std::string& images_path, int input_h, int input_w) {
    std::ifstream ifs(images_path);
    if (!ifs) {
        return tensorflow::errors::NotFound("Lables file ", images_path, " not found");
    }
    std::vector<std::string> result;
    result.clear();
    std::string line;
    while (std::getline(ifs, line)) {
        result.emplace_back(line);
    }
    if (result.size() < BATCH_SIZE) {
        LOG(INFO) << "Warnning: result.size() < BATCH_SIZE.";
    }
    auto t1 = std::chrono::system_clock::now();
    for (int b=0; b<BATCH_SIZE; b++) {
        cv::Mat image = cv::imread(result[b], 1);
        image.convertTo(image, CV_32FC3);
        image = image / 255;
        cv::resize(image, image, cv::Size(input_h, input_w));
        LOG(INFO) << "image_size: " << image.size().height << " " << image.size().width;
        cv::Mat transposeMat = image.t();
        auto tmap = input_tensor.tensor<float, 4>();
        const float* data = (float* )image.data;
        for (int y = 0; y < 224; ++y) {
            const float *dataRow = data + (y * 224 * 3);
            for (int x = 0; x < 224; ++x) {
                const float *dataPixel = dataRow + (x * 3);
                for (int c = 0; c < 3; ++c) {
                    const float *dataValue = dataPixel + c;
                    tmap(b, x, y, c) = *dataValue;
                }
            }
        }
    }
    auto t2 = std::chrono::system_clock::now();
    double t2t1 = static_cast<double >(std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count());
    LOG(INFO) << "mat to tensor cost time: " << t2t1;
    return Status::OK();
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Warning: the number of input parameters is wrong..." << std::endl;
        return -1;
    }
    auto t1 = std::chrono::system_clock::now();
    tensorflow::string model_path = argv[1];
    // TensorName pre-defined in python file, Need to extract values from tensors
    // add placeholder to modle input will easy there
    // images is not really true
    tensorflow::string image_path = "/home/xxx/DataSets/catvsdog/testImage.txt";
    tensorflow::string label_path = "/home/xxxDataSets/catvsdog/testLabel.txt";

    tensorflow::int32 input_width = 224;
    tensorflow::int32 input_height = 224;
    float input_mean = 0;
    float input_std = 255;
    tensorflow::string input_tensor_name = "input_x:0";
    tensorflow::string output_tensor_name = "softmax_linear/softmax_linear_1:0";
    bool self_test = false;
    tensorflow::string root_dir = "";

    std::vector<tensorflow::Flag> flag_list = {
            tensorflow::Flag("image", &image_path, "image to be processed"),
            tensorflow::Flag("graph", &model_path, "graph to be executed"),
            tensorflow::Flag("labels", &label_path, "name of file containing labels"),
            tensorflow::Flag("input_width", &input_width, "resize image to this width in pixels"),
            tensorflow::Flag("input_height", &input_height, "resize image to this height in pixels"),
            tensorflow::Flag("input_mean", &input_mean, "scale pixel values to this mean"),
            tensorflow::Flag("input_std", &input_std, "scale pixel values to this std deviation"),
            tensorflow::Flag("input_tensor_name", &input_tensor_name, "name of input layer"),
            tensorflow::Flag("output_tensor_name", &output_tensor_name, "name of output layer"),
            tensorflow::Flag("self_test", &self_test, "run a self test"),
            tensorflow::Flag("root_dir", &root_dir, "interpret image and graph file names relative to this directory"),
    };
    tensorflow::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
    const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
    if (!parse_result) {
        LOG(ERROR) << usage;
        return -1;
    }

    tensorflow::port::InitMain(argv[0], &argc, &argv);
    if (argc > 1) {
        LOG(ERROR) << "Unknow argument " << argv[1] << "\n" << usage;
        return -1;
    }

//    std::vector<Tensor> resized_tensors;
//    tensorflow::string input_path = tensorflow::io::JoinPath(root_dir, image_path);
//    LOG(INFO) << "the input_path is " << input_path;
//    Status read_tensor_status = ReadTensorFromImageFile(input_path, input_height, input_width, input_mean,
//                                                        input_std, &resized_tensors);
//    if (!read_tensor_status.ok()) {
//        LOG(INFO) << "read tensor status failed.";
//        return -1;
//    }
//    const Tensor& resized_tensor = resized_tensors[0];
//    LOG(INFO) << "resized_tensor.shape: " << resized_tensor.shape();

//    cv::Mat image = cv::imread(image_path, 1);
//    cv::resize(image, image, cv::Size(input_height, input_width));
//    tensorflow::Tensor resized_tensor;
//    Mat2tfTensor(image, 0, 255, resized_tensor);
//    LOG(INFO) << "resized_tensor.shape: " << resized_tensor.shape();

    tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, TensorShape({BATCH_SIZE, input_width, input_height, 3}));
    Status mat2tensorSattus = getMatTensor(input_tensor, image_path, input_height, input_width);
    if (!mat2tensorSattus.ok()) {
        LOG(INFO) << "Batch mat to tensor failed.";
        return -1;
    }

    // load and initialized the model
    std::unique_ptr<tensorflow::Session> session;
    tensorflow::string graph_path = tensorflow::io::JoinPath(root_dir, model_path);
    LOG(INFO) << "the graph_path is " << graph_path;
    Status load_graph_status = LoadGraph(graph_path, &session);
    if (!load_graph_status.ok()) {
        LOG(ERROR) << "Load graph failed.";
        return -1;
    }

    std::vector<Tensor> outputs;
    // std::pair<string, Tensor>inputInfo (input_tensor_name, resized_tensor);
    std::vector<std::pair<tensorflow::string, tensorflow::Tensor>> inputInfo = { {input_tensor_name, input_tensor} };
    Status run_status = session->Run({inputInfo}, {output_tensor_name}, {}, &outputs);
    if (!run_status.ok()) {
        LOG(ERROR) << "Running model failed: " << run_status;
        return -1;
    }

    // test outputs
    // tensorflow::Tensor outTensor = outputs[0];
    // const float* outTensorData = reinterpret_cast<const float*>(outTensor.tensor_data().data());
    // for (int i=0; i<outTensor.dim_size(0); i++) {
    //     LOG(INFO) << "outTensorData: " << outTensorData[i*2+0] << " " << outTensorData[i*2+1];
    // }
    // LOG(INFO) << "outTensor: " << outTensor.dim_size(0) << " " << outTensor.dim_size(1) << " " << outTensor.shape();

    if (self_test) {
        bool expected_matches;
        Status check_status = CheckTopLabel(outputs, 653, &expected_matches);
        if (!check_status.ok()) {
            LOG(ERROR) << "Running check failed: " << check_status;
            return -1;
        }
        if (!expected_matches) {
            LOG(ERROR) << "self test failed.";
            return -1;
        }
    }

    Status print_status = PrintTopLabels(outputs, label_path);
    if (!print_status.ok()) {
        LOG(ERROR) << "Running print failed: " << print_status;
        return -1;
    }
    return 0;
}
