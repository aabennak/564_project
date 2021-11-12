module MyDesign (
	input wire clk,
	input wire reset_b, 
	input wire dut_run, 
	input wire [15:0] sram_dut_read_data, 
	input wire [15:0] wmem_dut_read_data,
	output wire [15:0] dut_sram_read_address,
	output wire [15:0] dut_sram_write_address, 
	output wire [15:0] dut_sram_write_data,
	output wire [15:0] dut_wmem_read_address,
	output reg dut_busy, 
	output reg dut_sram_write_enable);

parameter [1:0] // synopsys enum states
   S0 = 3'b000, //reset state
   S1 = 3'b001, //calls for dimensions according to the saved addresses + clears computations from previous convolutions
   S2 = 3'b010, //recieves dimensions and saves them + calls for SRAM_input
   S3 = 3'b011; //No output stage
   S4 = 3'b100; //Output stage
   S5 = 3'b101; //
   S6 = 3'b110; //

// All accumulators
reg [3:0] row0_0acc, row0_1acc, row0_2acc,row0_3acc,row0_4acc,row0_5acc,row0_6acc,row0_7acc,row0_8acc,row0_9acc,row0_10acc,row0_11acc,row0_12acc,row0_13acc;
reg [3:0] row1_0acc, row1_1acc, row1_2acc,row1_3acc,row1_4acc,row1_5acc,row1_6acc,row1_7acc,row1_8acc,row1_9acc,row1_10acc,row1_11acc,row1_12acc,row1_13acc;
reg [3:0] row2_0acc, row2_1acc, row2_2acc,row2_3acc,row2_4acc,row2_5acc,row2_6acc,row2_7acc,row2_8acc,row2_9acc,row2_10acc,row2_11acc,row2_12acc,row2_13acc;

//State Regs
reg [2:0] current_state, next_state;

//
reg [11:0] SRAM_write_add; //write address pointer
reg [11:0] SRAM_read_add; // read address pointer 
reg [1:0] row_acc_sel; //selects which row will get the first attributes of the weight convolution... 00 for 1st, 01 for 2nd, 10 for 3rd
reg [1:0] input_dim; //00 is 10, 01 is 12, and 10 is 16
reg [4:0] input_count; //keeps track of how many inputs were there... resets with each convolution matrix
reg [8:0] theweights;

/*------- Sequential Logic ----*/
always@(posedge clock or negedge reset)
  if (!reset)   current_state <= S0;
  else  current_state <= next_state;

/* next state logic and output logic – combined so as to share state decode logic */
always@(current_state)
  begin /* defaults outputs to prevent latches */
     dut_sram_write_address = 16'h0; dut_sram_write_data= 16'h0; dut_sram_read_address= 16'h0; dut_sram_write_address= 16'h0; dut_sram_write_enable = 1'b0; dut_busy= 1'b0;
	 dut_wmem_read_address = 16'h0;
     case (current_state) // synopsys full_case parallel_case
        S0: begin //Reset state
            row0_0acc <= 4'b0; row0_1acc <= 4'b0 ; row0_2acc <= 4'b0 ;row0_3acc <= 4'b0 ;row0_4acc <= 4'b0 ;row0_5acc <= 4'b0 ;row0_6acc <= 4'b0 ;
			row0_7acc <= 4'b0 ;row0_8acc <= 4'b0 ;row0_9acc <= 4'b0 ;row0_10acc <= 4'b0 ;row0_11acc <= 4'b0 ;row0_12acc <= 4'b0 ;row0_13acc <= 4'b0 ;
			row1_0acc <= 4'b0 ; row1_1acc <= 4'b0 ; row1_2acc <= 4'b0 ;row1_3acc <= 4'b0 ;row1_4acc <= 4'b0 ;row1_5acc <= 4'b0 ;row1_6acc <= 4'b0 ;
			row1_7acc <= 4'b0 ;row1_8acc <= 4'b0 ;row1_9acc <= 4'b0 ;row1_10acc <= 4'b0 ;row1_11acc <= 4'b0 ;row1_12acc <= 4'b0 ;row1_13acc <= 4'b0 ;
			row2_0acc <= 4'b0 ; row2_1acc <= 4'b0 ; row2_2acc <= 4'b0 ;row2_3acc <= 4'b0 ;row2_4acc <= 4'b0 ;row2_5acc <= 4'b0 ;row2_6acc <= 4'b0 ;
			row2_7acc <= 4'b0 ;row2_8acc <= 4'b0 ;row2_9acc <= 4'b0 ;row2_10acc <= 4'b0 ;row2_11acc <= 4'b0 ;row2_12acc <= 4'b0 ;row2_13acc <= 4'b0 ;
			SRAM_write_add <= 16'h0; SRAM_read_add <= 16'h0; row_acc_sel <= 2'b00; input_dim <= 2'b00; input_count <= 5'b0; theweights <= 5'b0//Zeroing all internal regs
            if (dut_run == 1'b1) next_state = S1; else next_state = S0;
            end
        S1: begin //State that you call when computing a new comvolution matrix ... requesting the dimensions of the new input matrix
            row0_0acc <= 4'b0; row0_1acc <= 4'b0 ; row0_2acc <= 4'b0 ;row0_3acc <= 4'b0 ;row0_4acc <= 4'b0 ;row0_5acc <= 4'b0 ;row0_6acc <= 4'b0 ;
			row0_7acc <= 4'b0 ;row0_8acc <= 4'b0 ;row0_9acc <= 4'b0 ;row0_10acc <= 4'b0 ;row0_11acc <= 4'b0 ;row0_12acc <= 4'b0 ;row0_13acc <= 4'b0 ;
			row1_0acc <= 4'b0 ; row1_1acc <= 4'b0 ; row1_2acc <= 4'b0 ;row1_3acc <= 4'b0 ;row1_4acc <= 4'b0 ;row1_5acc <= 4'b0 ;row1_6acc <= 4'b0 ;
			row1_7acc <= 4'b0 ;row1_8acc <= 4'b0 ;row1_9acc <= 4'b0 ;row1_10acc <= 4'b0 ;row1_11acc <= 4'b0 ;row1_12acc <= 4'b0 ;row1_13acc <= 4'b0 ;
			row2_0acc <= 4'b0 ; row2_1acc <= 4'b0 ; row2_2acc <= 4'b0 ;row2_3acc <= 4'b0 ;row2_4acc <= 4'b0 ;row2_5acc <= 4'b0 ;row2_6acc <= 4'b0 ;
			row2_7acc <= 4'b0 ;row2_8acc <= 4'b0 ;row2_9acc <= 4'b0 ;row2_10acc <= 4'b0 ;row2_11acc <= 4'b0 ;row2_12acc <= 4'b0 ;row2_13acc <= 4'b0 ;
			SRAM_write_add <= SRAM_write_add; SRAM_read_add <= SRAM_read_add + 2'b10 ; //these conserve value... incrementing the reader pointer values
			row_acc_sel <= 2'b00; input_dim <= 2'b00; input_count <= 5'b0;  theweights <= 5'b0        //these reset
			dut_sram_read_address = SRAM_read_add; dut_wmem_read_address =  16'h0001;	 //requesting weight and input dimensions
			dut_busy = 1'b1;
            next_state = S2; 
            end
        S2 : begin
            row0_0acc <= 4'b0; row0_1acc <= 4'b0 ; row0_2acc <= 4'b0 ;row0_3acc <= 4'b0 ;row0_4acc <= 4'b0 ;row0_5acc <= 4'b0 ;row0_6acc <= 4'b0 ;
			row0_7acc <= 4'b0 ;row0_8acc <= 4'b0 ;row0_9acc <= 4'b0 ;row0_10acc <= 4'b0 ;row0_11acc <= 4'b0 ;row0_12acc <= 4'b0 ;row0_13acc <= 4'b0 ;
			row1_0acc <= 4'b0 ; row1_1acc <= 4'b0 ; row1_2acc <= 4'b0 ;row1_3acc <= 4'b0 ;row1_4acc <= 4'b0 ;row1_5acc <= 4'b0 ;row1_6acc <= 4'b0 ;
			row1_7acc <= 4'b0 ;row1_8acc <= 4'b0 ;row1_9acc <= 4'b0 ;row1_10acc <= 4'b0 ;row1_11acc <= 4'b0 ;row1_12acc <= 4'b0 ;row1_13acc <= 4'b0 ;
			row2_0acc <= 4'b0 ; row2_1acc <= 4'b0 ; row2_2acc <= 4'b0 ;row2_3acc <= 4'b0 ;row2_4acc <= 4'b0 ;row2_5acc <= 4'b0 ;row2_6acc <= 4'b0 ;
			row2_7acc <= 4'b0 ;row2_8acc <= 4'b0 ;row2_9acc <= 4'b0 ;row2_10acc <= 4'b0 ;row2_11acc <= 4'b0 ;row2_12acc <= 4'b0 ;row2_13acc <= 4'b0 ;
			SRAM_write_add <= SRAM_write_add; SRAM_read_add <= SRAM_read_add +1'b1 ; //these conserve value... incrementing the reader pointer values
			row_acc_sel <= 2'b00; input_count <= 5'b0;   //still zero since no input maxtrix was made
			if(sram_dut_read_data == 16'b10000) input_dim <= 2'b10 ; else if(sram_dut_read_data == 16'b1100) input_dim <= 2'b01; else input_dim <= 2'b00; //Recording dimension of the matrix
			theweights <= wmem_dut_read_data; 		//Recording the weight
			dut_sram_read_address = SRAM_read_add; 	 //requesting the next input
			dut_busy = 1'b1;
            if(sram_dut_read_data == 16'hFF) next_state = S5; else next_state = S3; 
            end
		S3 : begin //no writing just inputting
			if(input_count == 5'b00000) 
			begin //1st iteration
				row0_7acc <= (~(sram_dut_read_data[9]^theweights[8]) + ~(sram_dut_read_data[8]^theweights[7])) + ~(sram_dut_read_data[7]^theweights[6]);
				row0_6acc <= (~(sram_dut_read_data[8]^theweights[8]) + ~(sram_dut_read_data[7]^theweights[7])) + ~(sram_dut_read_data[6]^theweights[6]);
				row0_5acc <= (~(sram_dut_read_data[7]^theweights[8]) + ~(sram_dut_read_data[6]^theweights[7])) + ~(sram_dut_read_data[5]^theweights[6]);
				row0_4acc <= (~(sram_dut_read_data[6]^theweights[8]) + ~(sram_dut_read_data[5]^theweights[7])) + ~(sram_dut_read_data[4]^theweights[6]);
				row0_3acc <= (~(sram_dut_read_data[5]^theweights[8]) + ~(sram_dut_read_data[4]^theweights[7])) + ~(sram_dut_read_data[3]^theweights[6]);
				row0_2acc <= (~(sram_dut_read_data[4]^theweights[8]) + ~(sram_dut_read_data[3]^theweights[7])) + ~(sram_dut_read_data[2]^theweights[6]);
				row0_1acc <= (~(sram_dut_read_data[3]^theweights[8]) + ~(sram_dut_read_data[2]^theweights[7])) + ~(sram_dut_read_data[1]^theweights[6]);
				row0_0acc <= (~(sram_dut_read_data[2]^theweights[8]) + ~(sram_dut_read_data[1]^theweights[7])) + ~(sram_dut_read_data[0]^theweights[6]);
					if(input_dim == 2'b01 )begin //Case 12x12 matrix first iteration
						row0_9acc <= (~(sram_dut_read_data[11]^theweights[8]) + ~(sram_dut_read_data[10]^theweights[7])) + ~(sram_dut_read_data[9]^theweights[6]);
						row0_8acc <= (~(sram_dut_read_data[10]^theweights[8]) + ~(sram_dut_read_data[9]^theweights[7])) + ~(sram_dut_read_data[8]^theweights[6]);
						row0_10acc <= 4'b0 ;row0_11acc <= 4'b0 ;row0_12acc <= 4'b0 ;row0_13acc <= 4'b0 ;
						end
					else if(input_dim == 2'b10)begin //Case 16x 16 matrix first iteration
						row0_13acc <= (~(sram_dut_read_data[15]^theweights[8]) + ~(sram_dut_read_data[14]^theweights[7])) + ~(sram_dut_read_data[13]^theweights[6]); 
						row0_12acc <= (~(sram_dut_read_data[14]^theweights[8]) + ~(sram_dut_read_data[13]^theweights[7])) + ~(sram_dut_read_data[12]^theweights[6]);;
						row0_11acc <= (~(sram_dut_read_data[13]^theweights[8]) + ~(sram_dut_read_data[12]^theweights[7])) + ~(sram_dut_read_data[11]^theweights[6]);
						row0_10acc <= (~(sram_dut_read_data[12]^theweights[8]) + ~(sram_dut_read_data[11]^theweights[7])) + ~(sram_dut_read_data[10]^theweights[6]);
						row0_9acc <= (~(sram_dut_read_data[11]^theweights[8]) + ~(sram_dut_read_data[10]^theweights[7])) + ~(sram_dut_read_data[9]^theweights[6]);
						row0_8acc <= (~(sram_dut_read_data[10]^theweights[8]) + ~(sram_dut_read_data[9]^theweights[7])) + ~(sram_dut_read_data[8]^theweights[6]);
						end
					else begin //Case 10x10 first iteration
						row0_8acc <= 4'b0 ;row0_9acc <= 4'b0 ;row0_10acc <= 4'b0 ;row0_11acc <= 4'b0 ;row0_12acc <= 4'b0 ;row0_13acc <= 4'b0 ;
					end
					row1_0acc <= 4'b0 ; row1_1acc <= 4'b0 ; row1_2acc <= 4'b0 ;row1_3acc <= 4'b0 ;row1_4acc <= 4'b0 ;row1_5acc <= 4'b0 ;row1_6acc <= 4'b0 ;
					row1_7acc <= 4'b0 ;row1_8acc <= 4'b0 ;row1_9acc <= 4'b0 ;row1_10acc <= 4'b0 ;row1_11acc <= 4'b0 ;row1_12acc <= 4'b0 ;row1_13acc <= 4'b0 ;
					row2_0acc <= 4'b0 ; row2_1acc <= 4'b0 ; row2_2acc <= 4'b0 ;row2_3acc <= 4'b0 ;row2_4acc <= 4'b0 ;row2_5acc <= 4'b0 ;row2_6acc <= 4'b0 ;
					row2_7acc <= 4'b0 ;row2_8acc <= 4'b0 ;row2_9acc <= 4'b0 ;row2_10acc <= 4'b0 ;row2_11acc <= 4'b0 ;row2_12acc <= 4'b0 ;row2_13acc <= 4'b0 ;
				end
			else if(input_count == 5'b00001) 
				begin //2nd iteration
					row0_7acc <= (~(sram_dut_read_data[9]^theweights[5]) + ~(sram_dut_read_data[8]^theweights[4])) + (~(sram_dut_read_data[7]^theweights[3]) + row0_7acc);
					row0_6acc <= (~(sram_dut_read_data[8]^theweights[5]) + ~(sram_dut_read_data[7]^theweights[4])) + (~(sram_dut_read_data[6]^theweights[3]) + row0_6acc);
					row0_5acc <= (~(sram_dut_read_data[7]^theweights[5]) + ~(sram_dut_read_data[6]^theweights[4])) + (~(sram_dut_read_data[5]^theweights[3]) + row0_5acc);
					row0_4acc <= (~(sram_dut_read_data[6]^theweights[5]) + ~(sram_dut_read_data[5]^theweights[4])) + (~(sram_dut_read_data[4]^theweights[3]) + row0_4acc);
					row0_3acc <= (~(sram_dut_read_data[5]^theweights[5]) + ~(sram_dut_read_data[4]^theweights[4])) + (~(sram_dut_read_data[3]^theweights[3]) + row0_3acc);
					row0_2acc <= (~(sram_dut_read_data[4]^theweights[5]) + ~(sram_dut_read_data[3]^theweights[4])) + (~(sram_dut_read_data[2]^theweights[3]) + row0_2acc);
					row0_1acc <= (~(sram_dut_read_data[3]^theweights[5]) + ~(sram_dut_read_data[2]^theweights[4])) + (~(sram_dut_read_data[1]^theweights[3]) + row0_1acc);
					row0_0acc <= (~(sram_dut_read_data[2]^theweights[5]) + ~(sram_dut_read_data[1]^theweights[4])) + (~(sram_dut_read_data[0]^theweights[3]) + row0_0acc);
					row1_7acc <= (~(sram_dut_read_data[9]^theweights[8]) + ~(sram_dut_read_data[8]^theweights[7])) + ~(sram_dut_read_data[7]^theweights[6]);
					row1_6acc <= (~(sram_dut_read_data[8]^theweights[8]) + ~(sram_dut_read_data[7]^theweights[7])) + ~(sram_dut_read_data[6]^theweights[6]);
					row1_5acc <= (~(sram_dut_read_data[7]^theweights[8]) + ~(sram_dut_read_data[6]^theweights[7])) + ~(sram_dut_read_data[5]^theweights[6]);
					row1_4acc <= (~(sram_dut_read_data[6]^theweights[8]) + ~(sram_dut_read_data[5]^theweights[7])) + ~(sram_dut_read_data[4]^theweights[6]);
					row1_3acc <= (~(sram_dut_read_data[5]^theweights[8]) + ~(sram_dut_read_data[4]^theweights[7])) + ~(sram_dut_read_data[3]^theweights[6]);
					row1_2acc <= (~(sram_dut_read_data[4]^theweights[8]) + ~(sram_dut_read_data[3]^theweights[7])) + ~(sram_dut_read_data[2]^theweights[6]);
					row1_1acc <= (~(sram_dut_read_data[3]^theweights[8]) + ~(sram_dut_read_data[2]^theweights[7])) + ~(sram_dut_read_data[1]^theweights[6]);
					row1_0acc <= (~(sram_dut_read_data[2]^theweights[8]) + ~(sram_dut_read_data[1]^theweights[7])) + ~(sram_dut_read_data[0]^theweights[6]);
					if(input_dim == 2'b01)begin //Case 12x12 matrix second iteration
						row0_9acc <= (~(sram_dut_read_data[11]^theweights[5]) + ~(sram_dut_read_data[10]^theweights[4])) + (~(sram_dut_read_data[9]^theweights[3]) + row0_9acc);
						row0_8acc <= (~(sram_dut_read_data[10]^theweights[5]) + ~(sram_dut_read_data[9]^theweights[4])) + (~(sram_dut_read_data[8]^theweights[3]) + row0_8acc);
						row1_9acc <= (~(sram_dut_read_data[11]^theweights[8]) + ~(sram_dut_read_data[10]^theweights[7])) + ~(sram_dut_read_data[9]^theweights[6]);
						row1_8acc <= (~(sram_dut_read_data[10]^theweights[8]) + ~(sram_dut_read_data[9]^theweights[7])) + ~(sram_dut_read_data[8]^theweights[6]);
						row0_10acc <= 4'b0 ;row0_11acc <= 4'b0 ;row0_12acc <= 4'b0 ;row0_13acc <= 4'b0 ;
						row1_10acc <= 4'b0 ;row1_11acc <= 4'b0 ;row1_12acc <= 4'b0 ;row1_13acc <= 4'b0 ;
						end
					else if(input_dim == 2'b10)begin //Case 16x 16 matrix second iteration
						row0_13acc <= (~(sram_dut_read_data[15]^theweights[5]) + ~(sram_dut_read_data[14]^theweights[4])) + (~(sram_dut_read_data[13]^theweights[3]) + row0_13acc); 
						row0_12acc <= (~(sram_dut_read_data[14]^theweights[5]) + ~(sram_dut_read_data[13]^theweights[4])) + (~(sram_dut_read_data[12]^theweights[3]) + row0_12acc);
						row0_11acc <= (~(sram_dut_read_data[13]^theweights[5]) + ~(sram_dut_read_data[12]^theweights[4])) + (~(sram_dut_read_data[11]^theweights[3]) + row0_11acc);
						row0_10acc <= (~(sram_dut_read_data[12]^theweights[5]) + ~(sram_dut_read_data[11]^theweights[4])) + (~(sram_dut_read_data[10]^theweights[3]) + row0_10acc);
						row0_9acc <= (~(sram_dut_read_data[11]^theweights[5]) + ~(sram_dut_read_data[10]^theweights[4])) + (~(sram_dut_read_data[9]^theweights[3]) + row0_9acc);
						row0_8acc <= (~(sram_dut_read_data[10]^theweights[5]) + ~(sram_dut_read_data[9]^theweights[4])) + (~(sram_dut_read_data[8]^theweights[3]) + row0_8acc);
						row1_13acc <= (~(sram_dut_read_data[15]^theweights[8]) + ~(sram_dut_read_data[14]^theweights[7])) + ~(sram_dut_read_data[13]^theweights[6]); 
						row1_12acc <= (~(sram_dut_read_data[14]^theweights[8]) + ~(sram_dut_read_data[13]^theweights[7])) + ~(sram_dut_read_data[12]^theweights[6]);
						row1_11acc <= (~(sram_dut_read_data[13]^theweights[8]) + ~(sram_dut_read_data[12]^theweights[7])) + ~(sram_dut_read_data[11]^theweights[6]);
						row1_10acc <= (~(sram_dut_read_data[12]^theweights[8]) + ~(sram_dut_read_data[11]^theweights[7])) + ~(sram_dut_read_data[10]^theweights[6]);
						row1_9acc <= (~(sram_dut_read_data[11]^theweights[8]) + ~(sram_dut_read_data[10]^theweights[7])) + ~(sram_dut_read_data[9]^theweights[6]);
						row1_8acc <= (~(sram_dut_read_data[10]^theweights[8]) + ~(sram_dut_read_data[9]^theweights[7])) + ~(sram_dut_read_data[8]^theweights[6]);
						end
					else begin //Case 10x10 second iteration
						row0_8acc <= 4'b0 ;row0_9acc <= 4'b0 ;row0_10acc <= 4'b0 ;row0_11acc <= 4'b0 ;row0_12acc <= 4'b0 ;row0_13acc <= 4'b0 ;
						row1_8acc <= 4'b0 ;row1_9acc <= 4'b0 ;row1_10acc <= 4'b0 ;row1_11acc <= 4'b0 ;row1_12acc <= 4'b0 ;row1_13acc <= 4'b0 ;
					end
					row2_0acc <= 4'b0 ; row2_1acc <= 4'b0 ; row2_2acc <= 4'b0 ;row2_3acc <= 4'b0 ;row2_4acc <= 4'b0 ;row2_5acc <= 4'b0 ;row2_6acc <= 4'b0 ;
					row2_7acc <= 4'b0 ;row2_8acc <= 4'b0 ;row2_9acc <= 4'b0 ;row2_10acc <= 4'b0 ;row2_11acc <= 4'b0 ;row2_12acc <= 4'b0 ;row2_13acc <= 4'b0 ;
				end
			else 
				begin //3rd iteration base
					row0_7acc <= (~(sram_dut_read_data[9]^theweights[2]) + ~(sram_dut_read_data[8]^theweights[1])) + (~(sram_dut_read_data[7]^theweights[0]) + row0_7acc);
					row0_6acc <= (~(sram_dut_read_data[8]^theweights[2]) + ~(sram_dut_read_data[7]^theweights[1])) + (~(sram_dut_read_data[6]^theweights[0]) + row0_6acc);
					row0_5acc <= (~(sram_dut_read_data[7]^theweights[2]) + ~(sram_dut_read_data[6]^theweights[1])) + (~(sram_dut_read_data[5]^theweights[0]) + row0_5acc);
					row0_4acc <= (~(sram_dut_read_data[6]^theweights[2]) + ~(sram_dut_read_data[5]^theweights[1])) + (~(sram_dut_read_data[4]^theweights[0]) + row0_4acc);
					row0_3acc <= (~(sram_dut_read_data[5]^theweights[2]) + ~(sram_dut_read_data[4]^theweights[1])) + (~(sram_dut_read_data[3]^theweights[0]) + row0_3acc);
					row0_2acc <= (~(sram_dut_read_data[4]^theweights[2]) + ~(sram_dut_read_data[3]^theweights[1])) + (~(sram_dut_read_data[2]^theweights[0]) + row0_2acc);
					row0_1acc <= (~(sram_dut_read_data[3]^theweights[2]) + ~(sram_dut_read_data[2]^theweights[1])) + (~(sram_dut_read_data[1]^theweights[0]) + row0_1acc);
					row0_0acc <= (~(sram_dut_read_data[2]^theweights[2]) + ~(sram_dut_read_data[1]^theweights[1])) + (~(sram_dut_read_data[0]^theweights[0]) + row0_0acc);
					row1_7acc <= (~(sram_dut_read_data[9]^theweights[5]) + ~(sram_dut_read_data[8]^theweights[4])) + (~(sram_dut_read_data[7]^theweights[3]) + row1_7acc); 
					row1_6acc <= (~(sram_dut_read_data[8]^theweights[5]) + ~(sram_dut_read_data[7]^theweights[4])) + (~(sram_dut_read_data[6]^theweights[3]) + row1_6acc); 
					row1_5acc <= (~(sram_dut_read_data[7]^theweights[5]) + ~(sram_dut_read_data[6]^theweights[4])) + (~(sram_dut_read_data[5]^theweights[3]) + row1_5acc); 
					row1_4acc <= (~(sram_dut_read_data[6]^theweights[5]) + ~(sram_dut_read_data[5]^theweights[4])) + (~(sram_dut_read_data[4]^theweights[3]) + row1_4acc); 
					row1_3acc <= (~(sram_dut_read_data[5]^theweights[5]) + ~(sram_dut_read_data[4]^theweights[4])) + (~(sram_dut_read_data[3]^theweights[3]) + row1_3acc); 
					row1_2acc <= (~(sram_dut_read_data[4]^theweights[5]) + ~(sram_dut_read_data[3]^theweights[4])) + (~(sram_dut_read_data[2]^theweights[3]) + row1_2acc); 
					row1_1acc <= (~(sram_dut_read_data[3]^theweights[5]) + ~(sram_dut_read_data[2]^theweights[4])) + (~(sram_dut_read_data[1]^theweights[3]) + row1_1acc); 
					row1_0acc <= (~(sram_dut_read_data[2]^theweights[5]) + ~(sram_dut_read_data[1]^theweights[4])) + (~(sram_dut_read_data[0]^theweights[3]) + row1_0acc);
					row2_7acc <= (~(sram_dut_read_data[9]^theweights[8]) + ~(sram_dut_read_data[8]^theweights[7])) + ~(sram_dut_read_data[7]^theweights[6]);
					row2_6acc <= (~(sram_dut_read_data[8]^theweights[8]) + ~(sram_dut_read_data[7]^theweights[7])) + ~(sram_dut_read_data[6]^theweights[6]);
					row2_5acc <= (~(sram_dut_read_data[7]^theweights[8]) + ~(sram_dut_read_data[6]^theweights[7])) + ~(sram_dut_read_data[5]^theweights[6]);
					row2_4acc <= (~(sram_dut_read_data[6]^theweights[8]) + ~(sram_dut_read_data[5]^theweights[7])) + ~(sram_dut_read_data[4]^theweights[6]);
					row2_3acc <= (~(sram_dut_read_data[5]^theweights[8]) + ~(sram_dut_read_data[4]^theweights[7])) + ~(sram_dut_read_data[3]^theweights[6]);
					row2_2acc <= (~(sram_dut_read_data[4]^theweights[8]) + ~(sram_dut_read_data[3]^theweights[7])) + ~(sram_dut_read_data[2]^theweights[6]);
					row2_1acc <= (~(sram_dut_read_data[3]^theweights[8]) + ~(sram_dut_read_data[2]^theweights[7])) + ~(sram_dut_read_data[1]^theweights[6]);
					row2_0acc <= (~(sram_dut_read_data[2]^theweights[8]) + ~(sram_dut_read_data[1]^theweights[7])) + ~(sram_dut_read_data[0]^theweights[6]); 
					if(input_dim == 2'b01)begin //3rd iteration 12x 12 case
						row0_9acc <= (~(sram_dut_read_data[11]^theweights[2]) + ~(sram_dut_read_data[10]^theweights[1])) + (~(sram_dut_read_data[9]^theweights[0]) + row0_9acc);
						row0_8acc <= (~(sram_dut_read_data[10]^theweights[2]) + ~(sram_dut_read_data[9]^theweights[1])) + (~(sram_dut_read_data[8]^theweights[0]) + row0_8acc);
						row2_9acc <= (~(sram_dut_read_data[11]^theweights[8]) + ~(sram_dut_read_data[10]^theweights[7])) + ~(sram_dut_read_data[9]^theweights[6]);
						row2_8acc <= (~(sram_dut_read_data[10]^theweights[8]) + ~(sram_dut_read_data[9]^theweights[7])) + ~(sram_dut_read_data[8]^theweights[6]);
						row0_10acc <= 4'b0 ;row0_11acc <= 4'b0 ;row0_12acc <= 4'b0 ;row0_13acc <= 4'b0 ;
						row1_10acc <= 4'b0 ;row1_11acc <= 4'b0 ;row1_12acc <= 4'b0 ;row1_13acc <= 4'b0 ;
						row2_10acc <= 4'b0 ;row2_11acc <= 4'b0 ;row2_12acc <= 4'b0 ;row2_13acc <= 4'b0 ;
					end
					else if(input_dim == 2'b10)begin //3rd iteration 16 x 16 case
						row0_13acc <= (~(sram_dut_read_data[15]^theweights[2]) + ~(sram_dut_read_data[14]^theweights[1])) + (~(sram_dut_read_data[13]^theweights[0]) + row0_13acc); 
						row0_12acc <= (~(sram_dut_read_data[14]^theweights[2]) + ~(sram_dut_read_data[13]^theweights[1])) + (~(sram_dut_read_data[12]^theweights[0]) + row0_12acc);
						row0_11acc <= (~(sram_dut_read_data[13]^theweights[2]) + ~(sram_dut_read_data[12]^theweights[1])) + (~(sram_dut_read_data[11]^theweights[0]) + row0_11acc);
						row0_10acc <= (~(sram_dut_read_data[12]^theweights[2]) + ~(sram_dut_read_data[11]^theweights[1])) + (~(sram_dut_read_data[10]^theweights[0]) + row0_10acc);
						row0_9acc <= (~(sram_dut_read_data[11]^theweights[2]) + ~(sram_dut_read_data[10]^theweights[1])) + (~(sram_dut_read_data[9]^theweights[0]) + row0_9acc);
						row0_8acc <= (~(sram_dut_read_data[10]^theweights[2]) + ~(sram_dut_read_data[9]^theweights[1])) + (~(sram_dut_read_data[8]^theweights[0]) + row0_8acc);
						row1_13acc <= (~(sram_dut_read_data[15]^theweights[5]) + ~(sram_dut_read_data[14]^theweights[4])) + (~(sram_dut_read_data[13]^theweights[3]) + row1_13acc); 
						row1_12acc <= (~(sram_dut_read_data[14]^theweights[5]) + ~(sram_dut_read_data[13]^theweights[4])) + (~(sram_dut_read_data[12]^theweights[3]) + row1_12acc); 
						row1_11acc <= (~(sram_dut_read_data[13]^theweights[5]) + ~(sram_dut_read_data[12]^theweights[4])) + (~(sram_dut_read_data[11]^theweights[3]) + row1_11acc); 
						row1_10acc <= (~(sram_dut_read_data[12]^theweights[5]) + ~(sram_dut_read_data[11]^theweights[4])) + (~(sram_dut_read_data[10]^theweights[3]) + row1_10acc); 
						row2_13acc <= (~(sram_dut_read_data[15]^theweights[8]) + ~(sram_dut_read_data[14]^theweights[7])) + ~(sram_dut_read_data[13]^theweights[6]); 
						row2_12acc <= (~(sram_dut_read_data[14]^theweights[8]) + ~(sram_dut_read_data[13]^theweights[7])) + ~(sram_dut_read_data[12]^theweights[6]);;
						row2_11acc <= (~(sram_dut_read_data[13]^theweights[8]) + ~(sram_dut_read_data[12]^theweights[7])) + ~(sram_dut_read_data[11]^theweights[6]);
						row2_10acc <= (~(sram_dut_read_data[12]^theweights[8]) + ~(sram_dut_read_data[11]^theweights[7])) + ~(sram_dut_read_data[10]^theweights[6]);
						row2_9acc <= (~(sram_dut_read_data[11]^theweights[8]) + ~(sram_dut_read_data[10]^theweights[7])) + ~(sram_dut_read_data[9]^theweights[6]);
						row2_8acc <= (~(sram_dut_read_data[10]^theweights[8]) + ~(sram_dut_read_data[9]^theweights[7])) + ~(sram_dut_read_data[8]^theweights[6]);
						end
					else begin // 3rd iteration 10 x 10 case
					row0_8acc <= 4'b0 ;row0_9acc <= 4'b0 ;row0_10acc <= 4'b0 ;row0_11acc <= 4'b0 ;row0_12acc <= 4'b0 ;row0_13acc <= 4'b0 ;
					row1_8acc <= 4'b0 ;row1_9acc <= 4'b0 ;row1_10acc <= 4'b0 ;row1_11acc <= 4'b0 ;row1_12acc <= 4'b0 ;row1_13acc <= 4'b0 ;
					row2_8acc <= 4'b0 ;row2_9acc <= 4'b0 ;row2_10acc <= 4'b0 ;row2_11acc <= 4'b0 ;row2_12acc <= 4'b0 ;row2_13acc <= 4'b0 ;
					end
				end
			SRAM_write_add <= SRAM_write_add; SRAM_read_add <= SRAM_read_add + 1'b1 ; 
			row_acc_sel <= 2'b00 ; input_count <= input_count + 1'b1;   
			input_dim <= input_dim; 
			theweights <= theweights; 		//Weights would be same
			dut_sram_read_address = SRAM_read_add; 	 //requesting the next input
			dut_busy = 1'b1;
			if(input_count == 5'b10) next_state = S4; else next_state = S3;
            end//end of S3
        S4: begin //giving ouput and taking input
        	if(row_acc_sel == 2'b00) 
			begin //First row has to be outputted
				if(row0_13acc > 4) dut_sram_write_data[13] = 1'b1; else dut_sram_write_data[13] = 1'b0;
				if(row0_12acc > 4) dut_sram_write_data[12] = 1'b1; else dut_sram_write_data[12] = 1'b0;
				if(row0_11acc > 4) dut_sram_write_data[11] = 1'b1; else dut_sram_write_data[11] = 1'b0;
				if(row0_10cc > 4) dut_sram_write_data[10] = 1'b1; else dut_sram_write_data[10] = 1'b0;
				if(row0_9acc > 4)dut_sram_write_data[9] = 1'b1; else dut_sram_write_data[9] = 1'b0;
				if(row0_8acc > 4) dut_sram_write_data[8] = 1'b1; else dut_sram_write_data[8] = 1'b0;
				if(row0_7acc > 4) dut_sram_write_data[7] = 1'b1; else dut_sram_write_data[7] = 1'b0;
				if(row0_6acc > 4) dut_sram_write_data[6] = 1'b1; else dut_sram_write_data[6] = 1'b0;
				if(row0_5acc > 4) dut_sram_write_data[5] = 1'b1; else dut_sram_write_data[5] = 1'b0;
				if(row0_4acc > 4) dut_sram_write_data[4] = 1'b1; else dut_sram_write_data[4] = 1'b0;
				if(row0_3acc > 4) dut_sram_write_data[3] = 1'b1; else dut_sram_write_data[3] = 1'b0;
				if(row0_2acc > 4) dut_sram_write_data[2] = 1'b1; else dut_sram_write_data[2] = 1'b0;
				if(row0_1acc > 4) dut_sram_write_data[1] = 1'b1; else dut_sram_write_data[1] = 1'b0;
				if(row0_0acc > 4) dut_sram_write_data[0] = 1'b1; else dut_sram_write_data[0] = 1'b0;
				dut_sram_write_enable = 1'b1; dut_sram_write_address = SRAM_write_add; SRAM_write_add <= SRAM_write_add + 1'b1;
				
 				//1st ouputted base
				row0_7acc <= (~(sram_dut_read_data[9]^theweights[8]) + ~(sram_dut_read_data[8]^theweights[7])) + ~(sram_dut_read_data[7]^theweights[6]);
				row0_6acc <= (~(sram_dut_read_data[8]^theweights[8]) + ~(sram_dut_read_data[7]^theweights[7])) + ~(sram_dut_read_data[6]^theweights[6]);
				row0_5acc <= (~(sram_dut_read_data[7]^theweights[8]) + ~(sram_dut_read_data[6]^theweights[7])) + ~(sram_dut_read_data[5]^theweights[6]);
				row0_4acc <= (~(sram_dut_read_data[6]^theweights[8]) + ~(sram_dut_read_data[5]^theweights[7])) + ~(sram_dut_read_data[4]^theweights[6]);
				row0_3acc <= (~(sram_dut_read_data[5]^theweights[8]) + ~(sram_dut_read_data[4]^theweights[7])) + ~(sram_dut_read_data[3]^theweights[6]);
				row0_2acc <= (~(sram_dut_read_data[4]^theweights[8]) + ~(sram_dut_read_data[3]^theweights[7])) + ~(sram_dut_read_data[2]^theweights[6]);
				row0_1acc <= (~(sram_dut_read_data[3]^theweights[8]) + ~(sram_dut_read_data[2]^theweights[7])) + ~(sram_dut_read_data[1]^theweights[6]);
				row0_0acc <= (~(sram_dut_read_data[2]^theweights[8]) + ~(sram_dut_read_data[1]^theweights[7])) + ~(sram_dut_read_data[0]^theweights[6]);
				row1_7acc <= (~(sram_dut_read_data[9]^theweights[2]) + ~(sram_dut_read_data[8]^theweights[1])) + (~(sram_dut_read_data[7]^theweights[0]) + row1_7acc); 
				row1_6acc <= (~(sram_dut_read_data[8]^theweights[2]) + ~(sram_dut_read_data[7]^theweights[1])) + (~(sram_dut_read_data[6]^theweights[0]) + row1_6acc); 
				row1_5acc <= (~(sram_dut_read_data[7]^theweights[2]) + ~(sram_dut_read_data[6]^theweights[1])) + (~(sram_dut_read_data[5]^theweights[0]) + row1_5acc); 
				row1_4acc <= (~(sram_dut_read_data[6]^theweights[2]) + ~(sram_dut_read_data[5]^theweights[1])) + (~(sram_dut_read_data[4]^theweights[0]) + row1_4acc); 
				row1_3acc <= (~(sram_dut_read_data[5]^theweights[2]) + ~(sram_dut_read_data[4]^theweights[1])) + (~(sram_dut_read_data[3]^theweights[0]) + row1_3acc); 
				row1_2acc <= (~(sram_dut_read_data[4]^theweights[2]) + ~(sram_dut_read_data[3]^theweights[1])) + (~(sram_dut_read_data[2]^theweights[0]) + row1_2acc); 
				row1_1acc <= (~(sram_dut_read_data[3]^theweights[2]) + ~(sram_dut_read_data[2]^theweights[1])) + (~(sram_dut_read_data[1]^theweights[0]) + row1_1acc); 
				row1_0acc <= (~(sram_dut_read_data[2]^theweights[2]) + ~(sram_dut_read_data[1]^theweights[1])) + (~(sram_dut_read_data[0]^theweights[0]) + row1_0acc); 
				row2_7acc <= (~(sram_dut_read_data[9]^theweights[5]) + ~(sram_dut_read_data[8]^theweights[4])) + (~(sram_dut_read_data[7]^theweights[3])+ row2_7acc);
				row2_6acc <= (~(sram_dut_read_data[8]^theweights[5]) + ~(sram_dut_read_data[7]^theweights[4])) + (~(sram_dut_read_data[6]^theweights[3])+ row2_6acc);
				row2_5acc <= (~(sram_dut_read_data[7]^theweights[5]) + ~(sram_dut_read_data[6]^theweights[4])) + (~(sram_dut_read_data[5]^theweights[3])+ row2_5acc);
				row2_4acc <= (~(sram_dut_read_data[6]^theweights[5]) + ~(sram_dut_read_data[5]^theweights[4])) + (~(sram_dut_read_data[4]^theweights[3])+ row2_4acc);
				row2_3acc <= (~(sram_dut_read_data[5]^theweights[5]) + ~(sram_dut_read_data[4]^theweights[4])) + (~(sram_dut_read_data[3]^theweights[3])+ row2_3acc);
				row2_2acc <= (~(sram_dut_read_data[4]^theweights[5]) + ~(sram_dut_read_data[3]^theweights[4])) + (~(sram_dut_read_data[2]^theweights[3])+ row2_2acc);
				row2_1acc <= (~(sram_dut_read_data[3]^theweights[5]) + ~(sram_dut_read_data[2]^theweights[4])) + (~(sram_dut_read_data[1]^theweights[3])+ row2_1acc);
				row2_0acc <= (~(sram_dut_read_data[2]^theweights[5]) + ~(sram_dut_read_data[1]^theweights[4])) + (~(sram_dut_read_data[0]^theweights[3])+ row2_0acc);
					if(input_dim == 2'b01)begin
						row0_9acc <= (~(sram_dut_read_data[11]^theweights[8]) + ~(sram_dut_read_data[10]^theweights[7])) + ~(sram_dut_read_data[9]^theweights[6]);
						row0_8acc <= (~(sram_dut_read_data[10]^theweights[8]) + ~(sram_dut_read_data[9]^theweights[7])) + ~(sram_dut_read_data[8]^theweights[6]);
						row1_9acc <= (~(sram_dut_read_data[11]^theweights[2]) + ~(sram_dut_read_data[10]^theweights[1])) + (~(sram_dut_read_data[9]^theweights[0]) + row1_9acc); 
						row1_8acc <= (~(sram_dut_read_data[10]^theweights[2]) + ~(sram_dut_read_data[9]^theweights[1])) + (~(sram_dut_read_data[8]^theweights[0]) + row1_8acc); 
						row2_9acc <= (~(sram_dut_read_data[11]^theweights[5]) + ~(sram_dut_read_data[10]^theweights[4])) + (~(sram_dut_read_data[9]^theweights[3]) + row2_9acc);
						row2_8acc <= (~(sram_dut_read_data[10]^theweights[5]) + ~(sram_dut_read_data[9]^theweights[4])) + (~(sram_dut_read_data[8]^theweights[3])+ row2_8acc);
						row0_10acc <= 4'b0 ;row0_11acc <= 4'b0 ;row0_12acc <= 4'b0 ;row0_13acc <= 4'b0 ;
						row1_10acc <= 4'b0 ;row1_11acc <= 4'b0 ;row1_12acc <= 4'b0 ;row1_13acc <= 4'b0 ;
						row2_10acc <= 4'b0 ;row2_11acc <= 4'b0 ;row2_12acc <= 4'b0 ;row2_13acc <= 4'b0 ;
						end
					else if(input_dim == 2'b10)begin
						row0_13acc <= (~(sram_dut_read_data[15]^theweights[8]) + ~(sram_dut_read_data[14]^theweights[7])) + ~(sram_dut_read_data[13]^theweights[6]); 
						row0_12acc <= (~(sram_dut_read_data[14]^theweights[8]) + ~(sram_dut_read_data[13]^theweights[7])) + ~(sram_dut_read_data[12]^theweights[6]);
						row0_11acc <= (~(sram_dut_read_data[13]^theweights[8]) + ~(sram_dut_read_data[12]^theweights[7])) + ~(sram_dut_read_data[11]^theweights[6]);
						row0_10acc <= (~(sram_dut_read_data[12]^theweights[8]) + ~(sram_dut_read_data[11]^theweights[7])) + ~(sram_dut_read_data[10]^theweights[6]);
						row0_9acc <= (~(sram_dut_read_data[11]^theweights[8]) + ~(sram_dut_read_data[10]^theweights[7])) + ~(sram_dut_read_data[9]^theweights[6]);
						row0_8acc <= (~(sram_dut_read_data[10]^theweights[8]) + ~(sram_dut_read_data[9]^theweights[7])) + ~(sram_dut_read_data[8]^theweights[6]);
						row1_13acc <= (~(sram_dut_read_data[15]^theweights[2]) + ~(sram_dut_read_data[14]^theweights[1])) + (~(sram_dut_read_data[13]^theweights[0]) + row1_13acc); 
						row1_12acc <= (~(sram_dut_read_data[14]^theweights[2]) + ~(sram_dut_read_data[13]^theweights[1])) + (~(sram_dut_read_data[12]^theweights[0]) + row1_12acc); 
						row1_11acc <= (~(sram_dut_read_data[13]^theweights[2]) + ~(sram_dut_read_data[12]^theweights[1])) + (~(sram_dut_read_data[11]^theweights[0]) + row1_11acc); 
						row1_10acc <= (~(sram_dut_read_data[12]^theweights[2]) + ~(sram_dut_read_data[11]^theweights[1])) + (~(sram_dut_read_data[10]^theweights[0]) + row1_10acc); 
						row1_9acc <= (~(sram_dut_read_data[11]^theweights[2]) + ~(sram_dut_read_data[10]^theweights[1])) + (~(sram_dut_read_data[9]^theweights[0]) + row1_9acc); 
						row1_8acc <= (~(sram_dut_read_data[10]^theweights[2]) + ~(sram_dut_read_data[9]^theweights[1])) + (~(sram_dut_read_data[8]^theweights[0]) + row1_8acc); 
						row2_13acc <= (~(sram_dut_read_data[15]^theweights[5]) + ~(sram_dut_read_data[14]^theweights[4])) + (~(sram_dut_read_data[13]^theweights[3]) + row2_13acc); 
						row2_12acc <= (~(sram_dut_read_data[14]^theweights[5]) + ~(sram_dut_read_data[13]^theweights[4])) + (~(sram_dut_read_data[12]^theweights[3])+ row2_12acc);
						row2_11acc <= (~(sram_dut_read_data[13]^theweights[5]) + ~(sram_dut_read_data[12]^theweights[4])) + (~(sram_dut_read_data[11]^theweights[3]) + row2_11acc);
						row2_10acc <= (~(sram_dut_read_data[12]^theweights[5]) + ~(sram_dut_read_data[11]^theweights[4])) + (~(sram_dut_read_data[10]^theweights[3])+ row2_10acc);
						row2_9acc <= (~(sram_dut_read_data[11]^theweights[5]) + ~(sram_dut_read_data[10]^theweights[4])) + (~(sram_dut_read_data[9]^theweights[3])+ row2_9acc);
						row2_8acc <= (~(sram_dut_read_data[10]^theweights[5]) + ~(sram_dut_read_data[9]^theweights[4])) + (~(sram_dut_read_data[8]^theweights[3])+ row2_8acc);
						end
					else begin
						row0_8acc <= 4'b0 ;row0_9acc <= 4'b0 ;row0_10acc <= 4'b0 ;row0_11acc <= 4'b0 ;row0_12acc <= 4'b0 ;row0_13acc <= 4'b0 ;
						row1_8acc <= 4'b0 ;row1_9acc <= 4'b0 ;row1_10acc <= 4'b0 ;row1_11acc <= 4'b0 ;row1_12acc <= 4'b0 ;row1_13acc <= 4'b0 ;
						row2_8acc <= 4'b0 ;row2_9acc <= 4'b0 ;row2_10acc <= 4'b0 ;row2_11acc <= 4'b0 ;row2_12acc <= 4'b0 ;row2_13acc <= 4'b0 ;
					end
				end //end of 1st row output conditional
			if(row_acc_sel == 2'b01) 
				begin //seond row has to be outputted
				if(row1_13acc > 4) dut_sram_write_data[13] = 1'b1; else dut_sram_write_data[13] = 1'b0;
				if(row1_12acc > 4) dut_sram_write_data[12] = 1'b1; else dut_sram_write_data[12] = 1'b0;
				if(row1_11acc > 4) dut_sram_write_data[11] = 1'b1; else dut_sram_write_data[11] = 1'b0;
				if(row1_10cc > 4) dut_sram_write_data[10] = 1'b1; else dut_sram_write_data[10] = 1'b0;
				if(row1_9acc > 4)dut_sram_write_data[9] = 1'b1; else dut_sram_write_data[9] = 1'b0;
				if(row1_8acc > 4) dut_sram_write_data[8] = 1'b1; else dut_sram_write_data[8] = 1'b0;
				if(row1_7acc > 4) dut_sram_write_data[7] = 1'b1; else dut_sram_write_data[7] = 1'b0;
				if(row1_6acc > 4) dut_sram_write_data[6] = 1'b1; else dut_sram_write_data[6] = 1'b0;
				if(row1_5acc > 4) dut_sram_write_data[5] = 1'b1; else dut_sram_write_data[5] = 1'b0;
				if(row1_4acc > 4) dut_sram_write_data[4] = 1'b1; else dut_sram_write_data[4] = 1'b0;
				if(row1_3acc > 4) dut_sram_write_data[3] = 1'b1; else dut_sram_write_data[3] = 1'b0;
				if(row1_2acc > 4) dut_sram_write_data[2] = 1'b1; else dut_sram_write_data[2] = 1'b0;
				if(row1_1acc > 4) dut_sram_write_data[1] = 1'b1; else dut_sram_write_data[1] = 1'b0;
				if(row1_0acc > 4) dut_sram_write_data[0] = 1'b1; else dut_sram_write_data[0] = 1'b0;
				dut_sram_write_enable = 1'b1; dut_sram_write_address = SRAM_write_add; SRAM_write_add <= SRAM_write_add + 1'b1;			
 				//2nd row ouputted base
				row0_7acc <= (~(sram_dut_read_data[9]^theweights[5]) + ~(sram_dut_read_data[8]^theweights[4])) + (~(sram_dut_read_data[7]^theweights[3]) + row0_7acc);
				row0_6acc <= (~(sram_dut_read_data[8]^theweights[5]) + ~(sram_dut_read_data[7]^theweights[4])) + (~(sram_dut_read_data[6]^theweights[3]) + row0_6acc);
				row0_5acc <= (~(sram_dut_read_data[7]^theweights[5]) + ~(sram_dut_read_data[6]^theweights[4])) + (~(sram_dut_read_data[5]^theweights[3]) + row0_5acc);
				row0_4acc <= (~(sram_dut_read_data[6]^theweights[5]) + ~(sram_dut_read_data[5]^theweights[4])) + (~(sram_dut_read_data[4]^theweights[3]) + row0_4acc);
				row0_3acc <= (~(sram_dut_read_data[5]^theweights[5]) + ~(sram_dut_read_data[4]^theweights[4])) + (~(sram_dut_read_data[3]^theweights[3]) + row0_3acc);
				row0_2acc <= (~(sram_dut_read_data[4]^theweights[5]) + ~(sram_dut_read_data[3]^theweights[4])) + (~(sram_dut_read_data[2]^theweights[3]) + row0_2acc);
				row0_1acc <= (~(sram_dut_read_data[3]^theweights[5]) + ~(sram_dut_read_data[2]^theweights[4])) + (~(sram_dut_read_data[1]^theweights[3]) + row0_1acc);
				row0_0acc <= (~(sram_dut_read_data[2]^theweights[5]) + ~(sram_dut_read_data[1]^theweights[4])) + (~(sram_dut_read_data[0]^theweights[3]) + row0_0acc);
				row1_7acc <= (~(sram_dut_read_data[9]^theweights[8]) + ~(sram_dut_read_data[8]^theweights[7])) + ~(sram_dut_read_data[7]^theweights[6]); 
				row1_6acc <= (~(sram_dut_read_data[8]^theweights[8]) + ~(sram_dut_read_data[7]^theweights[7])) + ~(sram_dut_read_data[6]^theweights[6]); 
				row1_5acc <= (~(sram_dut_read_data[7]^theweights[8]) + ~(sram_dut_read_data[6]^theweights[7])) + ~(sram_dut_read_data[5]^theweights[6]); 
				row1_4acc <= (~(sram_dut_read_data[6]^theweights[8]) + ~(sram_dut_read_data[5]^theweights[7])) + ~(sram_dut_read_data[4]^theweights[6]); 
				row1_3acc <= (~(sram_dut_read_data[5]^theweights[8]) + ~(sram_dut_read_data[4]^theweights[7])) + ~(sram_dut_read_data[3]^theweights[6]); 
				row1_2acc <= (~(sram_dut_read_data[4]^theweights[8]) + ~(sram_dut_read_data[3]^theweights[7])) + ~(sram_dut_read_data[2]^theweights[6]); 
				row1_1acc <= (~(sram_dut_read_data[3]^theweights[8]) + ~(sram_dut_read_data[2]^theweights[7])) + ~(sram_dut_read_data[1]^theweights[6]); 
				row1_0acc <= (~(sram_dut_read_data[2]^theweights[8]) + ~(sram_dut_read_data[1]^theweights[7])) + ~(sram_dut_read_data[0]^theweights[6]); 
				row2_7acc <= (~(sram_dut_read_data[9]^theweights[2]) + ~(sram_dut_read_data[8]^theweights[1])) + (~(sram_dut_read_data[7]^theweights[0]) + row2_7acc);
				row2_6acc <= (~(sram_dut_read_data[8]^theweights[2]) + ~(sram_dut_read_data[7]^theweights[1])) + (~(sram_dut_read_data[6]^theweights[0]) + row2_6acc);
				row2_5acc <= (~(sram_dut_read_data[7]^theweights[2]) + ~(sram_dut_read_data[6]^theweights[1])) + (~(sram_dut_read_data[5]^theweights[0]) + row2_5acc);
				row2_4acc <= (~(sram_dut_read_data[6]^theweights[2]) + ~(sram_dut_read_data[5]^theweights[1])) + (~(sram_dut_read_data[4]^theweights[0]) + row2_4acc);
				row2_3acc <= (~(sram_dut_read_data[5]^theweights[2]) + ~(sram_dut_read_data[4]^theweights[1])) + (~(sram_dut_read_data[3]^theweights[0]) + row2_3acc);
				row2_2acc <= (~(sram_dut_read_data[4]^theweights[2]) + ~(sram_dut_read_data[3]^theweights[1])) + (~(sram_dut_read_data[2]^theweights[0]) + row2_2acc);
				row2_1acc <= (~(sram_dut_read_data[3]^theweights[2]) + ~(sram_dut_read_data[2]^theweights[1])) + (~(sram_dut_read_data[1]^theweights[0]) + row2_1acc);
				row2_0acc <= (~(sram_dut_read_data[2]^theweights[2]) + ~(sram_dut_read_data[1]^theweights[1])) + (~(sram_dut_read_data[0]^theweights[0]) + row2_0acc);
					if(input_dim == 2'b01)begin
						row0_9acc <= (~(sram_dut_read_data[11]^theweights[5]) + ~(sram_dut_read_data[10]^theweights[4])) + (~(sram_dut_read_data[9]^theweights[3]) + row0_9acc);
						row0_8acc <= (~(sram_dut_read_data[10]^theweights[5]) + ~(sram_dut_read_data[9]^theweights[4])) + (~(sram_dut_read_data[8]^theweights[3])+ row0_8acc);
						row1_9acc <= (~(sram_dut_read_data[11]^theweights[8]) + ~(sram_dut_read_data[10]^theweights[7])) + ~(sram_dut_read_data[9]^theweights[6]); 
						row1_8acc <= (~(sram_dut_read_data[10]^theweights[8]) + ~(sram_dut_read_data[9]^theweights[7])) + ~(sram_dut_read_data[8]^theweights[6]) ; 
						row2_9acc <= (~(sram_dut_read_data[11]^theweights[2]) + ~(sram_dut_read_data[10]^theweights[1])) + (~(sram_dut_read_data[9]^theweights[0]) + row2_9acc);
						row2_8acc <= (~(sram_dut_read_data[10]^theweights[2]) + ~(sram_dut_read_data[9]^theweights[1])) + (~(sram_dut_read_data[8]^theweights[0]) + row2_8acc);
						row0_10acc <= 4'b0 ;row0_11acc <= 4'b0 ;row0_12acc <= 4'b0 ;row0_13acc <= 4'b0 ;
						row1_10acc <= 4'b0 ;row1_11acc <= 4'b0 ;row1_12acc <= 4'b0 ;row1_13acc <= 4'b0 ;
						row2_10acc <= 4'b0 ;row2_11acc <= 4'b0 ;row2_12acc <= 4'b0 ;row2_13acc <= 4'b0 ;
						end
					else if(input_dim == 2'b10)begin
						row0_13acc <= (~(sram_dut_read_data[15]^theweights[5]) + ~(sram_dut_read_data[14]^theweights[4])) + (~(sram_dut_read_data[13]^theweights[3]) + row0_13acc); 
						row0_12acc <= (~(sram_dut_read_data[14]^theweights[5]) + ~(sram_dut_read_data[13]^theweights[4])) + (~(sram_dut_read_data[12]^theweights[3]) + row0_12acc);
						row0_11acc <= (~(sram_dut_read_data[13]^theweights[5]) + ~(sram_dut_read_data[12]^theweights[4])) + (~(sram_dut_read_data[11]^theweights[3]) + row0_11acc);
						row0_10acc <= (~(sram_dut_read_data[12]^theweights[5]) + ~(sram_dut_read_data[11]^theweights[4])) + (~(sram_dut_read_data[10]^theweights[3]) + row0_10acc);
						row0_9acc <= (~(sram_dut_read_data[11]^theweights[5]) + ~(sram_dut_read_data[10]^theweights[4])) + (~(sram_dut_read_data[9]^theweights[3]) + row0_9acc);
						row0_8acc <= (~(sram_dut_read_data[10]^theweights[5]) + ~(sram_dut_read_data[9]^theweights[4])) + (~(sram_dut_read_data[8]^theweights[3]) + row0_8acc);
						row1_13acc <= (~(sram_dut_read_data[15]^theweights[8]) + ~(sram_dut_read_data[14]^theweights[7])) + ~(sram_dut_read_data[13]^theweights[6]); 
						row1_12acc <= (~(sram_dut_read_data[14]^theweights[8]) + ~(sram_dut_read_data[13]^theweights[7])) + ~(sram_dut_read_data[12]^theweights[6]); 
						row1_11acc <= (~(sram_dut_read_data[13]^theweights[8]) + ~(sram_dut_read_data[12]^theweights[7])) + ~(sram_dut_read_data[11]^theweights[6]); 
						row1_10acc <= (~(sram_dut_read_data[12]^theweights[8]) + ~(sram_dut_read_data[11]^theweights[7])) + ~(sram_dut_read_data[10]^theweights[6]); 
						row1_9acc <= (~(sram_dut_read_data[11]^theweights[8]) + ~(sram_dut_read_data[10]^theweights[7])) + ~(sram_dut_read_data[9]^theweights[6]); 
						row1_8acc <= (~(sram_dut_read_data[10]^theweights[8]) + ~(sram_dut_read_data[9]^theweights[7])) + ~(sram_dut_read_data[8]^theweights[6]); 
						row2_13acc <= (~(sram_dut_read_data[15]^theweights[2]) + ~(sram_dut_read_data[14]^theweights[1])) + (~(sram_dut_read_data[13]^theweights[0] + row2_13acc)); 
						row2_12acc <= (~(sram_dut_read_data[14]^theweights[2]) + ~(sram_dut_read_data[13]^theweights[1])) + (~(sram_dut_read_data[12]^theweights[0]+ row2_12acc));;
						row2_11acc <= (~(sram_dut_read_data[13]^theweights[2]) + ~(sram_dut_read_data[12]^theweights[1])) + (~(sram_dut_read_data[11]^theweights[0] + row2_11acc));
						row2_10acc <= (~(sram_dut_read_data[12]^theweights[2]) + ~(sram_dut_read_data[11]^theweights[1])) + (~(sram_dut_read_data[10]^theweights[0]+ row2_10acc));
						row2_9acc <= (~(sram_dut_read_data[11]^theweights[2]) + ~(sram_dut_read_data[10]^theweights[1])) + (~(sram_dut_read_data[9]^theweights[0]+ row2_9acc));
						row2_8acc <= (~(sram_dut_read_data[10]^theweights[2]) + ~(sram_dut_read_data[9]^theweights[1])) + (~(sram_dut_read_data[8]^theweights[0]+ row2_8acc));
						end
				end //end of second output conditional
				if(row_acc_sel == 2'b10) 
				begin //third row has to be outputted
				if(row2_13acc > 4) dut_sram_write_data[13] = 1'b1; else dut_sram_write_data[13] = 1'b0;
				if(row2_12acc > 4) dut_sram_write_data[12] = 1'b1; else dut_sram_write_data[12] = 1'b0;
				if(row2_11acc > 4) dut_sram_write_data[11] = 1'b1; else dut_sram_write_data[11] = 1'b0;
				if(row2_10cc > 4) dut_sram_write_data[10] = 1'b1; else dut_sram_write_data[10] = 1'b0;
				if(row2_9acc > 4)dut_sram_write_data[9] = 1'b1; else dut_sram_write_data[9] = 1'b0;
				if(row2_8acc > 4) dut_sram_write_data[8] = 1'b1; else dut_sram_write_data[8] = 1'b0;
				if(row2_7acc > 4) dut_sram_write_data[7] = 1'b1; else dut_sram_write_data[7] = 1'b0;
				if(row2_6acc > 4) dut_sram_write_data[6] = 1'b1; else dut_sram_write_data[6] = 1'b0;
				if(row2_5acc > 4) dut_sram_write_data[5] = 1'b1; else dut_sram_write_data[5] = 1'b0;
				if(row2_4acc > 4) dut_sram_write_data[4] = 1'b1; else dut_sram_write_data[4] = 1'b0;
				if(row2_3acc > 4) dut_sram_write_data[3] = 1'b1; else dut_sram_write_data[3] = 1'b0;
				if(row2_2acc > 4) dut_sram_write_data[2] = 1'b1; else dut_sram_write_data[2] = 1'b0;
				if(row2_1acc > 4) dut_sram_write_data[1] = 1'b1; else dut_sram_write_data[1] = 1'b0;
				if(row2_0acc > 4) dut_sram_write_data[0] = 1'b1; else dut_sram_write_data[0] = 1'b0;
				dut_sram_write_enable = 1'b1; dut_sram_write_address = SRAM_write_add; SRAM_write_add <= SRAM_write_add + 1'b1;			
 				//Third ouputted base
				row0_7acc <= (~(sram_dut_read_data[9]^theweights[2]) + ~(sram_dut_read_data[8]^theweights[1])) + (~(sram_dut_read_data[7]^theweights[0]) + row0_7acc);
				row0_6acc <= (~(sram_dut_read_data[8]^theweights[2]) + ~(sram_dut_read_data[7]^theweights[1])) + (~(sram_dut_read_data[6]^theweights[0]) + row0_6acc);
				row0_5acc <= (~(sram_dut_read_data[7]^theweights[2]) + ~(sram_dut_read_data[6]^theweights[1])) + (~(sram_dut_read_data[5]^theweights[0]) + row0_5acc);
				row0_4acc <= (~(sram_dut_read_data[6]^theweights[2]) + ~(sram_dut_read_data[5]^theweights[1])) + (~(sram_dut_read_data[4]^theweights[0]) + row0_4acc);
				row0_3acc <= (~(sram_dut_read_data[5]^theweights[2]) + ~(sram_dut_read_data[4]^theweights[1])) + (~(sram_dut_read_data[3]^theweights[0]) + row0_3acc);
				row0_2acc <= (~(sram_dut_read_data[4]^theweights[2]) + ~(sram_dut_read_data[3]^theweights[1])) + (~(sram_dut_read_data[2]^theweights[0]) + row0_2acc);
				row0_1acc <= (~(sram_dut_read_data[3]^theweights[2]) + ~(sram_dut_read_data[2]^theweights[1])) + (~(sram_dut_read_data[1]^theweights[0]) + row0_1acc);
				row0_0acc <= (~(sram_dut_read_data[2]^theweights[2]) + ~(sram_dut_read_data[1]^theweights[1])) + (~(sram_dut_read_data[0]^theweights[0]) + row0_0acc);
				row1_7acc <= (~(sram_dut_read_data[9]^theweights[5]) + ~(sram_dut_read_data[8]^theweights[4])) + (~(sram_dut_read_data[7]^theweights[3]) + row1_7acc); 
				row1_6acc <= (~(sram_dut_read_data[8]^theweights[5]) + ~(sram_dut_read_data[7]^theweights[4])) + (~(sram_dut_read_data[6]^theweights[3]) + row1_6acc); 
				row1_5acc <= (~(sram_dut_read_data[7]^theweights[5]) + ~(sram_dut_read_data[6]^theweights[4])) + (~(sram_dut_read_data[5]^theweights[3]) + row1_5acc); 
				row1_4acc <= (~(sram_dut_read_data[6]^theweights[5]) + ~(sram_dut_read_data[5]^theweights[4])) + (~(sram_dut_read_data[4]^theweights[3]) + row1_4acc); 
				row1_3acc <= (~(sram_dut_read_data[5]^theweights[5]) + ~(sram_dut_read_data[4]^theweights[4])) + (~(sram_dut_read_data[3]^theweights[3]) + row1_3acc); 
				row1_2acc <= (~(sram_dut_read_data[4]^theweights[5]) + ~(sram_dut_read_data[3]^theweights[4])) + (~(sram_dut_read_data[2]^theweights[3]) + row1_2acc); 
				row1_1acc <= (~(sram_dut_read_data[3]^theweights[5]) + ~(sram_dut_read_data[2]^theweights[4])) + (~(sram_dut_read_data[1]^theweights[3]) + row1_1acc); 
				row1_0acc <= (~(sram_dut_read_data[2]^theweights[5]) + ~(sram_dut_read_data[1]^theweights[4])) + (~(sram_dut_read_data[0]^theweights[3]) + row1_0acc); 
				row2_7acc <= (~(sram_dut_read_data[9]^theweights[8]) + ~(sram_dut_read_data[8]^theweights[7])) + ~(sram_dut_read_data[7]^theweights[6]);
				row2_6acc <= (~(sram_dut_read_data[8]^theweights[8]) + ~(sram_dut_read_data[7]^theweights[7])) + ~(sram_dut_read_data[6]^theweights[6]);
				row2_5acc <= (~(sram_dut_read_data[7]^theweights[8]) + ~(sram_dut_read_data[6]^theweights[7])) + ~(sram_dut_read_data[5]^theweights[6]);
				row2_4acc <= (~(sram_dut_read_data[6]^theweights[8]) + ~(sram_dut_read_data[5]^theweights[7])) + ~(sram_dut_read_data[4]^theweights[6]);
				row2_3acc <= (~(sram_dut_read_data[5]^theweights[8]) + ~(sram_dut_read_data[4]^theweights[7])) + ~(sram_dut_read_data[3]^theweights[6]);
				row2_2acc <= (~(sram_dut_read_data[4]^theweights[8]) + ~(sram_dut_read_data[3]^theweights[7])) + ~(sram_dut_read_data[2]^theweights[6]);
				row2_1acc <= (~(sram_dut_read_data[3]^theweights[8]) + ~(sram_dut_read_data[2]^theweights[7])) + ~(sram_dut_read_data[1]^theweights[6]);
				row2_0acc <= (~(sram_dut_read_data[2]^theweights[8]) + ~(sram_dut_read_data[1]^theweights[7])) + ~(sram_dut_read_data[0]^theweights[6]);
					if(input_dim == 2'b01)begin
						row0_9acc <= (~(sram_dut_read_data[11]^theweights[2]) + ~(sram_dut_read_data[10]^theweights[1])) + (~(sram_dut_read_data[9]^theweights[0]) + row0_9acc);
						row0_8acc <= (~(sram_dut_read_data[10]^theweights[2]) + ~(sram_dut_read_data[9]^theweights[1])) + (~(sram_dut_read_data[8]^theweights[0])+ row0_8acc);
						row1_9acc <= (~(sram_dut_read_data[11]^theweights[5]) + ~(sram_dut_read_data[10]^theweights[4])) + (~(sram_dut_read_data[9]^theweights[3]) + row1_9acc); 
						row1_8acc <= (~(sram_dut_read_data[10]^theweights[5]) + ~(sram_dut_read_data[9]^theweights[4])) + (~(sram_dut_read_data[8]^theweights[3]) + row1_8acc); 
						row2_9acc <= (~(sram_dut_read_data[11]^theweights[8]) + ~(sram_dut_read_data[10]^theweights[7])) + ~(sram_dut_read_data[9]^theweights[6]);
						row2_8acc <= (~(sram_dut_read_data[10]^theweights[8]) + ~(sram_dut_read_data[9]^theweights[7])) + ~(sram_dut_read_data[8]^theweights[6]);
						row0_10acc <= 4'b0 ;row0_11acc <= 4'b0 ;row0_12acc <= 4'b0 ;row0_13acc <= 4'b0 ;
						row1_10acc <= 4'b0 ;row1_11acc <= 4'b0 ;row1_12acc <= 4'b0 ;row1_13acc <= 4'b0 ;
						row2_10acc <= 4'b0 ;row2_11acc <= 4'b0 ;row2_12acc <= 4'b0 ;row2_13acc <= 4'b0 ;
						end
					else if(input_dim == 2'b10)begin
						row0_13acc <= (~(sram_dut_read_data[15]^theweights[2]) + ~(sram_dut_read_data[14]^theweights[1])) + (~(sram_dut_read_data[13]^theweights[0]) + row0_13acc); 
						row0_12acc <= (~(sram_dut_read_data[14]^theweights[2]) + ~(sram_dut_read_data[13]^theweights[1])) + (~(sram_dut_read_data[12]^theweights[0]) + row0_12acc);
						row0_11acc <= (~(sram_dut_read_data[13]^theweights[2]) + ~(sram_dut_read_data[12]^theweights[1])) + (~(sram_dut_read_data[11]^theweights[0]) + row0_11acc);
						row0_10acc <= (~(sram_dut_read_data[12]^theweights[2]) + ~(sram_dut_read_data[11]^theweights[1])) + (~(sram_dut_read_data[10]^theweights[0]) + row0_10acc);
						row0_9acc <= (~(sram_dut_read_data[11]^theweights[2]) + ~(sram_dut_read_data[10]^theweights[1])) + (~(sram_dut_read_data[9]^theweights[0]) + row0_9acc);
						row0_8acc <= (~(sram_dut_read_data[10]^theweights[2]) + ~(sram_dut_read_data[9]^theweights[1])) + (~(sram_dut_read_data[8]^theweights[0])+ row0_8acc);
						row1_13acc <= (~(sram_dut_read_data[15]^theweights[5]) + ~(sram_dut_read_data[14]^theweights[4])) + (~(sram_dut_read_data[13]^theweights[3]) + row1_13acc); 
						row1_12acc <= (~(sram_dut_read_data[14]^theweights[5]) + ~(sram_dut_read_data[13]^theweights[4])) + (~(sram_dut_read_data[12]^theweights[3]) + row1_12acc); 
						row1_11acc <= (~(sram_dut_read_data[13]^theweights[5]) + ~(sram_dut_read_data[12]^theweights[4])) + (~(sram_dut_read_data[11]^theweights[3]) + row1_11acc); 
						row1_10acc <= (~(sram_dut_read_data[12]^theweights[5]) + ~(sram_dut_read_data[11]^theweights[4])) + (~(sram_dut_read_data[10]^theweights[3]) + row1_10acc); 
						row1_9acc <= (~(sram_dut_read_data[11]^theweights[5]) + ~(sram_dut_read_data[10]^theweights[4])) + (~(sram_dut_read_data[9]^theweights[3]) + row1_9acc); 
						row1_8acc <= (~(sram_dut_read_data[10]^theweights[5]) + ~(sram_dut_read_data[9]^theweights[4])) + (~(sram_dut_read_data[8]^theweights[3]) + row1_8acc); 
						row2_13acc <= (~(sram_dut_read_data[15]^theweights[8]) + ~(sram_dut_read_data[14]^theweights[7])) + ~(sram_dut_read_data[13]^theweights[6]); 
						row2_12acc <= (~(sram_dut_read_data[14]^theweights[8]) + ~(sram_dut_read_data[13]^theweights[7])) + ~(sram_dut_read_data[12]^theweights[6]);
						row2_11acc <= (~(sram_dut_read_data[13]^theweights[8]) + ~(sram_dut_read_data[12]^theweights[7])) + ~(sram_dut_read_data[11]^theweights[6]);
						row2_10acc <= (~(sram_dut_read_data[12]^theweights[8]) + ~(sram_dut_read_data[11]^theweights[7])) + ~(sram_dut_read_data[10]^theweights[6]);
						row2_9acc <= (~(sram_dut_read_data[11]^theweights[8]) + ~(sram_dut_read_data[10]^theweights[7])) + ~(sram_dut_read_data[9]^theweights[6]);
						row2_8acc <= (~(sram_dut_read_data[10]^theweights[8]) + ~(sram_dut_read_data[9]^theweights[7])) + ~(sram_dut_read_data[8]^theweights[6]);
						end
				end //end of third output conditional
			SRAM_write_add <= SRAM_write_add + 1 ; SRAM_read_add <= SRAM_read_add +1'b1 ; 
			if(row_acc_sel == 10) row_acc_sel <= 0; else row_acc_sel <= 2'b00 +1 ; 
			input_count <= input_count + 1'b1;   
			input_dim <= input_dim; 
			theweights <= theweights; 		//Weights would be same
			dut_sram_read_address = SRAM_read_add; 	 //requesting the next input
			dut_busy = 1'b1;
			if((input_count == 5'b1010 & input_dim == 2'b00) | (input_count == 5'b1100 & input_dim == 2'b01) | (input_count == 5'b10000 & input_dim == 2'b10) )  next_state = S1; else next_state = S4;
            end// end of S4
        S5: begin
        	row0_0acc <= 4'b0; row0_1acc <= 4'b0 ; row0_2acc <= 4'b0 ;row0_3acc <= 4'b0 ;row0_4acc <= 4'b0 ;row0_5acc <= 4'b0 ;row0_6acc <= 4'b0 ;
			row0_7acc <= 4'b0 ;row0_8acc <= 4'b0 ;row0_9acc <= 4'b0 ;row0_10acc <= 4'b0 ;row0_11acc <= 4'b0 ;row0_12acc <= 4'b0 ;row0_13acc <= 4'b0 ;
			row1_0acc <= 4'b0 ; row1_1acc <= 4'b0 ; row1_2acc <= 4'b0 ;row1_3acc <= 4'b0 ;row1_4acc <= 4'b0 ;row1_5acc <= 4'b0 ;row1_6acc <= 4'b0 ;
			row1_7acc <= 4'b0 ;row1_8acc <= 4'b0 ;row1_9acc <= 4'b0 ;row1_10acc <= 4'b0 ;row1_11acc <= 4'b0 ;row1_12acc <= 4'b0 ;row1_13acc <= 4'b0 ;
			row2_0acc <= 4'b0 ; row2_1acc <= 4'b0 ; row2_2acc <= 4'b0 ;row2_3acc <= 4'b0 ;row2_4acc <= 4'b0 ;row2_5acc <= 4'b0 ;row2_6acc <= 4'b0 ;
			row2_7acc <= 4'b0 ;row2_8acc <= 4'b0 ;row2_9acc <= 4'b0 ;row2_10acc <= 4'b0 ;row2_11acc <= 4'b0 ;row2_12acc <= 4'b0 ;row2_13acc <= 4'b0 ;
			SRAM_write_add <= 16'h0; SRAM_read_add <= 16'h0; row_acc_sel <= 2'b00; input_dim <= 2'b00; input_count <= 5'b0; theweights <= 5'b0;//Zeroing all internal regs
            next_state = S5;
        end
       default : next_state = S0;
    endcase
  end

endmodule