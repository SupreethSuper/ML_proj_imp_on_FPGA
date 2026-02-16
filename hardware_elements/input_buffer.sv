// 
//  * @module input_buffer
//  * @brief Dual-port RAM buffer for input data storage and retrieval
//  * 
//  * This module implements a synchronous input buffer with independent read and write ports.
//  * It supports asynchronous reset and provides registered output data.
//  * 
//  * @param BUFFER_DATA_WIDTH - Width of data words stored in buffer (default: DATA_WIDTH)
//  * @param BUFFER_ADDR_WIDTH - Width of address bus (default: ADDR_WIDTH)
//  * 
//  * @port clk - System clock (posedge triggered)
//  * @port rst_n - Asynchronous active-low reset
//  * @port wr_en - Write enable signal
//  * @port rd_en - Read enable signal
//  * @port wr_addr - Write address
//  * @port rd_addr - Read address
//  * @port wr_data - Data to write into buffer
//  * @port rd_data - Data read from buffer (registered output)
//  * 
//  * @behavior
//  *   - On rst_n LOW: rd_data is asynchronously cleared to zero
//  *   - On clk POSEDGE: Output data is updated based on read operations
//  *   - Supports simultaneous independent read and write operations
//  
 `include "GLOBAL_PARAMS.vh"
 
 module input_buffer #(
    parameter BUFFER_DATA_WIDTH = DATA_WIDTH,
    parameter BUFFER_ADDR_WIDTH = ADDR_WIDTH
 )(
    input logic clk,
    input logic rst_n,
    input logic wr_en,
    input logic rd_en,

    
    input logic [ BUFFER_ADDR_WIDTH - 1 : 0 ] wr_addr,
    input logic [ BUFFER_ADDR_WIDTH - 1 : 0 ] rd_addr,


    input  logic signed [ BUFFER_DATA_WIDTH - 1 : 0 ] wr_data,

    output logic signed [ BUFFER_DATA_WIDTH - 1 : 0 ] rd_data

 );

    logic signed [ 0 : BUFFER_DATA_WIDTH - 1 ] mem [ (2 * BUFFER_ADDR_WIDTH) - 1 : 0 ];
    // logic collision_flag;
    // logic collision_clear;


    always_ff @( posedge clk or negedge rst_n ) begin : write_to_memory
        

        //async reset, triiger at rst low
        if(!rst_n) begin
            
            // collision_flag <= 1'b0;
        end
        else begin
            //to write data to mem at that address
            if(wr_en) begin
                mem[wr_addr] <= wr_data;
            end

            //exception case. But for safety, we read out first, then write in
            if(wr_en && rd_en) begin
                mem[wr_addr] <= wr_data; //written after 2 clk cycle delay

                // collision_flag <= 1'b1;
                // if(!collision_clear) begin
                //     
                //     collision_flag <= 1'b0;
                // end
            end
        end


    end

    //===============================================================================================




    //===========================================GAP ON PURPOSE=========================================




    //====================================================================================================
    always_ff @( posedge clk or negedge rst_n ) begin : write_to_output
        

        //async reset, triiger at rst low
        if(!rst_n) begin
            rd_data <= '0;
            rd_data <= mem[rd_addr];
        end
        else begin

            //to read data from mem at that address
            if(rd_en) begin
                rd_data <= mem[rd_addr];
            end
                //exception case. But for safety, we read out first, then write in
            if(wr_en && rd_en) begin
                // collision_clear <= 1'b1;
                
                // collision_clear <= 1'b0;
                 rd_data <= mem[rd_addr];
            end
        end


    end    
	
 
 
 endmodule