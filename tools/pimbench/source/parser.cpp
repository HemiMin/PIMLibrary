#include "parser.h"
#include <boost/program_options.hpp>
#include "elt_perf.h"
#include "gemm_perf.h"

//using namespace boost::program_options;
int Parser::print_help()
{
    std::cout << desc;
    return -1;
}
Parser::Parser()
{
    desc.add_options()("help,h", "Help screen")(
        "device,d", boost::program_options::value<int>(&device_id)->default_value(0),
        "sets the device for PIM execution in multi gpu scenario (default : 0)")(
        "platform,p", boost::program_options::value<string>(&platform)->default_value("hip"),
        "set the platform for PIM execution (default : hip)")(
        "pscn",
        boost::program_options::value<string>(&precision)->default_value("fp16", "sets the precision for PIM operations (default : fp16)"))(
        "operation,o", boost::program_options::value<string>(&operation), "indicates which operation is to be run on PIM")(
        "num_batch,n", boost::program_options::value<int>(&num_batch), "set the number of batch dimension")(
        "channel,c", boost::program_options::value<int>(&num_channels), "sets the number of channel dimension")(
        "i_h", boost::program_options::value<int>(&input_height), "sets the input height dimension")("i_w", boost::program_options::value<int>(&input_width),
                                                                             "sets the input width")(
        "o_h", boost::program_options::value<int>(&output_height), "sets the output height")("o_w", boost::program_options::value<int>(&output_width),
                                                                     "sets the output width")(
        "order", boost::program_options::value<string>(&order), "order(i_x_w / w_x_i) : sets the order for gemm operation")(
        "activation,a", boost::program_options::value<string>(&act_function)->default_value("relu"),
        "act (relu / none) : set the activation function for gemm operation (default : relu)")(
        "bias,b", boost::program_options::value<int>(&has_bias)->default_value(1),
        "bias (0 / 1) : sets if the gemm operation has bias addition (default : 1)")(
        "num_iter,i", boost::program_options::value<int>(&num_iter)->default_value(2),
        "sets the number of iterations to be executed for a particualr operation. (default : 2)");
}

boost::program_options::variables_map Parser::parse_args(int argc, char *argv[])
{
  boost::program_options::variables_map vm;
    try {
        store(parse_command_line(argc, argv, desc), vm);
        notify(vm);
    } catch (const boost::program_options::error &ex) {
        std::cerr << ex.what() << '\n';
    }
    return vm;
}

bool Parser::check_validity()
{
    if (platform != "hip" && platform != "opencl") {
        std::cout << "platform : " << platform << std::endl;
        DLOG(ERROR) << "invalid platform provided\n";
        return false;
    }

    if (precision != "fp16") {
        DLOG(ERROR) << "PIM doesnt support precision compute provided\n";
        return false;
    }

    if (operation == "") {
        DLOG(ERROR) << "op (operation) argument is missing\n";
        return false;
    }

    if (num_batch == -1 || num_channels == -1 || input_height == -1 || input_width == -1) {
        DLOG(ERROR) << "input information is missing\n";
        return false;
    }

    if (operation == "gemm") {
        if (order == "") {
            DLOG(ERROR) << "compute order is missing\n";
            return false;
        }
        if (output_height == -1 || output_width == -1) {
            DLOG(ERROR) << "output information is missing\n";
            return false;
        }
    }
    return true;
}
