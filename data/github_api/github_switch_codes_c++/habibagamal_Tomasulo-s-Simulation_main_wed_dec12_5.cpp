////////To get label in label vector --> check while reading instructions --> if it's not a known instruction --> then it's label --> get location of label using count then decrement count
////////In BEQ and JUMP detect label when parsing
////////Branch predictor function added. Explained at end of file.
#include <iostream>
#include <vector>
#include <queue>
#include <string>
#include <fstream>
#include <map>
#include <iomanip>

using namespace std;
#define memSize 32768
#define robSize 6
#define rfSize 8
#define BITPRED 2

int clk = 0;
int pc = 0;
int instEnd = 0;
int instCount = 0;
int head = 0;
int tail = 0;
int bitsPred = 0;
int countCommit = 0;
int countBranch = 0;
int countMissBranch = 0;
int instIndex = 0;
bool done = false;

struct reservationStationEntry
{
    bool busy;
    string op;
    int Vj, Vk;
    int Qj, Qk;
    int dest;
    int addr;
    int instAddr;
    int instIndex;
};

struct parsedInst
{
    string inst;
    string op;
    int rd;
    int rs1;
    int rs2;
    int imm;
    int addr;
    int result;
    int pcAddr;
    int index;
};

struct ROBEntry
{
    bool busy;
    int dest;
    int value;
    bool ready;
    parsedInst inst;
    int pcAddr;
};

struct ClockTracing
{
    int count = 0;
    int fetch;
    int Issued;
    int Executed;
    int Written;
    int commit;
    parsedInst par;
    bool flushed;
};

struct predictorEntry
{
    long adr;
    int state;
};

struct instStatusEntry
{
    parsedInst inst;
    int issued, executed, written, committed;
};

string memory[memSize];

map<string, int> labels;

queue<parsedInst> instrBuffer; //should be of size 4

ROBEntry ROB[robSize]; //size 6 because 6 ROB entries

int registerStatus[rfSize]; //size 8 because 8 registers
int registerFile[rfSize];

//TODO: I think fe haga mehtaga tetsalah hena -ali
reservationStationEntry loadStation[2];  //lw
reservationStationEntry storeStation[2]; //sw

reservationStationEntry jumpStation[3]; //JMP, JALR, RET
reservationStationEntry beqStation[2];  //beq
reservationStationEntry addStation[3];  //add, addi, sub
reservationStationEntry nandStation[1]; //nand
reservationStationEntry multStation[2]; //mult

vector<predictorEntry> predictionBuffer;
vector<instStatusEntry> instStatus;
vector<ClockTracing> trace;

void readFile(string filename);
void listInput();
parsedInst parser(string instruction, int addr);
bool isNumber(string s);
bool branchPredictor(int bitsPrediction, bool mode, int address, bool Bresult, int imm);
void issue(reservationStationEntry R[], int size, parsedInst I, int traceIndex);
bool issueHazards();
void InitializeRegisters();
void inputStream();
void write();
void fetch();
void issueCall();
void execution();
void resetReservation();
void commit();
void flush();
void resetInstrBuffer();
void resetROB();
void resetRAT();
void outputValues();
void outputTable();

int main()
{
    int inType, type;
    string filename;
    string instruction;
    string data;

    registerFile[0] = 0;
    cout << "Choose 1 if input from file, \nChoose 2 if input from input stream, \nChoose 3 if from a list" << endl;
    cin >> inType;
//    inType = 1; // FOR EASE OF TESTING
    if (inType == 1)
    {
        cout << "Enter filename" << endl;
        cin >> filename;
        readFile(filename);
//        readFile("test.txt"); // FOR EASE OF TESTING
    }
    else if (inType == 2)
    {
        inputStream();
    }
    else if (inType == 3)
    {
        listInput();
    }

    //////////////////// START TOMASULO ////////////////////
    clk = 1;
    InitializeRegisters();
    resetROB();
    resetInstrBuffer();
    resetReservation();
    resetRAT();

    while (instrBuffer.size() > 0 || pc <= instEnd || !done)
    {
        fetch();
        //if(!issueHazards())
        issueCall();
        execution();
        write();
        commit();
        clk++;
    }

    outputTable();
    outputValues();

    return 0;
}

void readFile(string filename)
{
    ifstream inp;
    int instAddr, dataAddr, count = 0;
    inp.open(filename);
    if (inp.is_open())
    {
        string temp;
        getline(inp, temp);
        while (!inp.eof())
        {
            if (temp.find(".text:") != string::npos)
            {
                getline(inp, temp);
                if (!inp.eof())
                {
                    temp.erase(temp.begin());
                    instAddr = stoi(temp);
                    count = instAddr;
                    pc = instAddr;
                    getline(inp, temp);
                }
            }
            else if (temp.find(".data:") != string::npos)
            {
                getline(inp, temp);
                while (!inp.eof())
                {
                    int i = 1;
                    string dataAddress = "";
                    while (temp[i] != ' ')
                    {
                        dataAddress += temp[i];
                        i++;
                    }
                    dataAddr = stoi(dataAddress);
                    i++;
                    string data = "";
                    for (i; i < temp.length(); i++)
                    {
                        data += temp[i];
                    }
                    if (dataAddr < instAddr || dataAddr > instEnd)
                        memory[dataAddr] = data;
                    getline(inp, temp);
                }
            }
            else if (temp[temp.length() - 1] == ':') //Labels
            {
                temp.erase(temp.end() - 1);
                labels[temp] = count;
                getline(inp, temp);
                // pair <string, int> p;
                // p.first = temp;
                // p.second = count;
                // labels.push_back(p);
            }
            else
            {
                memory[count] = temp;
                count++;
                getline(inp, temp);
            }
        }
        instEnd = count - 1;
    }
    else
    {
        cout << "error opening file" << endl;
    }
}

parsedInst parser(string instruction, int addr)
{
    string temp = "";
    parsedInst Parsed;
    Parsed.addr = addr;
    Parsed.inst = instruction;
    int i = 0;
    while (instruction[i] != ' ')
    {
        temp = temp + instruction[i];
        i++;
    }
    Parsed.op = temp;
    temp = "";
    if ((Parsed.op == "ADD") || (Parsed.op == "SUB") || (Parsed.op == "NAND") || (Parsed.op == "MUL"))
    {
        i = i + 2;
        while (instruction[i] != ',')
        {
            temp = temp + instruction[i];
            i++;
        }
        Parsed.rd = stoi(temp);
        i = i + 3;
        temp = "";
        while (instruction[i] != ',')
        {
            temp = temp + instruction[i];
            i++;
        }
        Parsed.rs1 = stoi(temp);
        i = i + 3;
        temp = "";
        for (i; i < instruction.length(); i++)
        {
            temp = temp + instruction[i];
        }
        Parsed.rs2 = stoi(temp);
    }
    else if ((Parsed.op == "ADDI") || ((Parsed.op == "LW")))
    {
        i = i + 2;
        while (instruction[i] != ',')
        {
            temp = temp + instruction[i];
            i++;
        }
        Parsed.rd = stoi(temp);
        i = i + 3;
        temp = "";
        while (instruction[i] != ',')
        {
            temp = temp + instruction[i];
            i++;
        }
        Parsed.rs1 = stoi(temp);
        i = i + 2;
        temp = "";
        for (i; i < instruction.length(); i++)
        {
            temp = temp + instruction[i];
        }
        Parsed.imm = stoi(temp);
        Parsed.rs2 = -1;
    }
    else if ((Parsed.op == "BEQ") || (Parsed.op == "SW"))
    {
        i = i + 2;
        while (instruction[i] != ',')
        {
            temp = temp + instruction[i];
            i++;
        }
        Parsed.rs1 = stoi(temp);
        i = i + 3;
        temp = "";
        while (instruction[i] != ',')
        {
            temp = temp + instruction[i];
            i++;
        }
        Parsed.rs2 = stoi(temp);
        i = i + 2;
        temp = "";
        for (i; i < instruction.length(); i++)
        {
            temp = temp + instruction[i];
        }

        if (isNumber(temp))
        {
            Parsed.imm = stoi(temp);
        }
        else
        {
            Parsed.imm = labels[temp] - addr - 1;
        }
        Parsed.rd = -1;
    }
    else if ((Parsed.op == "JMP"))
    {
        i = i + 1;
        for (i; i < instruction.length(); i++)
        {
            temp = temp + instruction[i];
        }

        if (isNumber(temp))
        {
            Parsed.imm = stoi(temp);
        }
        else
        {
            Parsed.imm = labels[temp] - addr - 1;
        }
        Parsed.rs2 = -1;
        Parsed.rs1 = -1;
        Parsed.rd = -1;
    }
    else if ((Parsed.op == "JALR"))
    {
        i = i + 2;
        while (instruction[i] != ',')
        {
            temp = temp + instruction[i];
            i++;
        }
        Parsed.rd = stoi(temp);
        i = i + 3;
        temp = "";
        for (i; i < instruction.length(); i++)
        {
            temp = temp + instruction[i];
        }
        Parsed.rs1 = stoi(temp);
        Parsed.rs2 = -1;
    }
    else if ((Parsed.op == "RET"))
    {
        i = i + 2;
        for (i; i < instruction.length(); i++)
        {
            temp = temp + instruction[i];
        }
        Parsed.rs1 = stoi(temp);
        Parsed.rs2 = -1;
        Parsed.rd = -1;
    }
    return Parsed;
}

bool isNumber(string s)
{
    bool flag = true;

    for (int i = 0; i < s.length(); i++)
    {
        if (!(isdigit(s[i])))
        {
            flag = false;
        }
    }

    return (flag);
}

void listInput()
{
    int instAddr, type, op, rd, rs1, rs2, imm, count, dataAddr;
    string data;
    string instruction;

    cout << "Choose starting address\n";
    cin >> instAddr;
    pc = instAddr;
    cout << "Choose Type:\nChoose 1 if load/store\nChoose 2 if unconditional branch\nChoose 3 if conditional branch\nChoose 4 if call/return\nChoose 5 if arithmetic\nChoose 0 if you're done\n";
    cin >> type;
    queue<string> instructions;
    while (type != 0)
    {
        instCount++;
        switch (type)
        {
        case (1): // load/store
        {
            cout << "Choose instruction:\nChoose 1 if LW\nChoose 2 if SW\n";
            cin >> op;

            switch (op)
            {
            case (1): //LW
            {
                cout << "Choose destination register number from 0 to 7" << endl;
                cin >> rd;
                cout << "Choose source register 1 number from 0 to 7" << endl;
                cin >> rs1;
                cout << "Input immediate" << endl;
                cin >> imm;
                instruction = "LW R" + to_string(rd) + ", R" + to_string(rs1) + ", " + to_string(imm);
                instructions.push(instruction);
            }
            break;
            case (2): //SW
            {
                cout << "Choose source register 1 number from 0 to 7" << endl;
                cin >> rs1;
                cout << "Choose source register 2 number from 0 to 7" << endl;
                cin >> rs2;
                cout << "Input immediate" << endl;
                cin >> imm;
                instruction = "SW R" + to_string(rs1) + ", R" + to_string(rs2) + ", " + to_string(imm);
                instructions.push(instruction);
            }
            break;
            }
        }
        break;
        case (2): //unconditional branch //JMP
        {
            cout << "Input immediate" << endl;
            cin >> imm;
            instruction = "JMP " + to_string(imm);
            instructions.push(instruction);
        }
        break;
        case (3): //conditional branch //BEQ
        {
            cout << "Choose source register 1 number from 0 to 7" << endl;
            cin >> rs1;
            cout << "Choose source register 2 number from 0 to 7" << endl;
            cin >> rs2;
            cout << "Input immediate" << endl;
            cin >> imm;
            instruction = "BEQ R" + to_string(rs1) + ", R" + to_string(rs2) + ", " + to_string(imm);
            instructions.push(instruction);
        }
        break;
        case (4): // Call/Return
        {
            cout << "Choose instruction:\nChoose 1 if JALR\nChoose 2 if RET\n";
            cin >> op;
            switch (op)
            {
            case (1): //JALR
            {
                cout << "Choose destination register number from 0 to 7" << endl;
                cin >> rd;
                cout << "Choose source register 1 number from 0 to 7" << endl;
                cin >> rs1;
                instruction = "JALR R" + to_string(rd) + ", R" + to_string(rs1);
                instructions.push(instruction);
            }
            break;
            case (2):
            {
                cout << "Choose source register 1 number from 0 to 7" << endl;
                cin >> rs1;
                instruction = "RET R" + to_string(rs1);
                instructions.push(instruction);
            }
            break;
            }
        }
        break;
        case (5): //Arithmetic
        {
            cout << "Choose instruction:\nChoose 1 if ADD\nChoose 2 if SUB\nChoose 3 if ADDI\nChoose 4 if NAND\nChoose 5 if MUL";
            cin >> op;
            switch (op)
            {
            case (1): //ADD
            {
                cout << "Choose destination register number from 0 to 7" << endl;
                cin >> rd;
                cout << "Choose source register 1 number from 0 to 7" << endl;
                cin >> rs1;
                cout << "Choose source register 2 number from 0 to 7" << endl;
                cin >> rs2;
                instruction = "ADD R" + to_string(rd) + ", R" + to_string(rs1) + ", R" + to_string(rs2);
                instructions.push(instruction);
            }
            break;
            case (2): //SUB
            {
                cout << "Choose destination register number from 0 to 7" << endl;
                cin >> rd;
                cout << "Choose source register 1 number from 0 to 7" << endl;
                cin >> rs1;
                cout << "Choose source register 2 number from 0 to 7" << endl;
                cin >> rs2;
                instruction = "SUB R" + to_string(rd) + ", R" + to_string(rs1) + ", R" + to_string(rs2);
                instructions.push(instruction);
            }
            break;
            case (3): //ADDI
            {
                cout << "Choose destination register number from 0 to 7" << endl;
                cin >> rd;
                cout << "Choose source register 1 number from 0 to 7" << endl;
                cin >> rs1;
                cout << "Input immediate" << endl;
                cin >> imm;
                instruction = "ADDI R" + to_string(rd) + ", R" + to_string(rs1) + ", " + to_string(imm);
                instructions.push(instruction);
            }
            break;
            case (4): //NAND
            {
                cout << "Choose destination register number from 0 to 7" << endl;
                cin >> rd;
                cout << "Choose source register 1 number from 0 to 7" << endl;
                cin >> rs1;
                cout << "Choose source register 2 number from 0 to 7" << endl;
                cin >> rs2;
                instruction = "NAND R" + to_string(rd) + ", R" + to_string(rs1) + ", R" + to_string(rs2);
                instructions.push(instruction);
            }
            break;
            case (5): //MUL
            {
                cout << "Choose destination register number from 0 to 7" << endl;
                cin >> rd;
                cout << "Choose source register 1 number from 0 to 7" << endl;
                cin >> rs1;
                cout << "Choose source register 2 number from 0 to 7" << endl;
                cin >> rs2;
                instruction = "MUL R" + to_string(rd) + ", R" + to_string(rs1) + ", R" + to_string(rs2);
                instructions.push(instruction);
            }
            break;
            }
            break;
        }
        }
        cout << "Choose Type:\nChoose 1 if load/store\nChoose 2 if unconditional branch\nChoose 3 if conditional branch\nChoose 4 if call/return\nChoose 5 if arithmetic\nChoose 0 if you're done\n";
        cin >> type;
    }
    count = instAddr;
    while (instructions.size() > 0)
    {
        string temp = instructions.front();
        instructions.pop();
        cout << temp << endl;
        memory[count] = temp;
        count++;
    }
    int option;
    instEnd = count - 1;
    cout << "If you'd like to enter data enter 1, else enter 0";
    cin >> option;
    if (option == 1)
    {
        cout << "Enter Data. Enter -1 if you're done:" << endl;
        cin >> data;
        while (data != "-1")
        {
            cout << "Enter address of data." << endl;
            cin >> dataAddr;
            while ((dataAddr <= instEnd) && (dataAddr != -1) && ((dataAddr) >= instAddr))
            {
                cout << "Invalid address. Enter address again\n";
                cin >> dataAddr;
            }
            count = dataAddr;
            memory[count] = data;
            count++;
            cout << "Enter Data. Enter -1 if you're done:" << endl;
            cin >> data;
        }
    }
}

bool issueHazards()
{
    parsedInst p;
    if (instrBuffer.size() > 0)
    {
        p = instrBuffer.front();
        if (head == tail && ROB[tail + 1].busy) //ROB is full
        {
            return 1;
        }
        if ((p.op == "ADD") || (p.op == "SUB") || (p.op == "ADDI")) //add station is full and instruction needs it
        {
            if ((addStation[0].busy == 1) && (addStation[1].busy == 1) && (addStation[2].busy == 1))
                return 1;
            else
                return 0;
        }
        else if ((p.op == "JAL") || (p.op == "JALR") || (p.op == "RET")) //jump station is full and instruction needs it
        {
            if ((jumpStation[0].busy == 1) && (jumpStation[1].busy == 1) && (jumpStation[2].busy == 1))
                return 1;
            else
                return 0;
        }
        else if (p.op == "LW") //load station is full and instruction is load
        {
            if ((loadStation[0].busy == 1) && (loadStation[1].busy == 1))
                return 1;
            else
                return 0;
        }
        else if (p.op == "SW") //store station is full and instruction is sw
        {
            if ((storeStation[0].busy == 1) && (storeStation[1].busy == 1))
                return 1;
            else
                return 0;
        }
        else if (p.op == "BEQ") //branch station is full and instruction is beq
        {
            if ((beqStation[0].busy == 1) && (beqStation[1].busy == 1))
                return 1;
            else
                return 0;
        }
        else if (p.op == "NAND") //nand station is full and instruction is nand
        {
            if (nandStation[0].busy == 1)
                return 1;
            else
                return 0;
        }
        else if (p.op == "MUL") //mul station is full and instruction is mul
        {
            if ((multStation[0].busy == 1) && (multStation[0].busy == 1))
                return 1;
            else
                return 0;
        }
        else
            return 0;
    }
    return 0;
}

void inputStream()
{
    int instAddr;
    int count;
    string instruction;
    int dataAddr;
    string data;
    
    cout << "Enter address of instruction. Enter -1 if you're done" << endl;
    cin >> instAddr;
    pc = instAddr;
    string temp;
    getline (cin, temp);
    count = instAddr;
    if (instAddr != -1)
    {
        cout << "Enter instruction. Enter 0 if you're done:" << endl;
        getline (cin , instruction);
        while (instruction != "0")
        {
            memory[count] = instruction;
            getline (cin , instruction);
            count++;
        }
    }
    instEnd = count - 1;
    cout << "Enter address of data. Enter -1 if you're done" << endl;
    cin >> dataAddr;
    while (dataAddr != -1)
    {
        while (((dataAddr >= instAddr) && (dataAddr <= instEnd)) && (dataAddr != -1))
        {
            cout << "Invalid address. Enter address again\n";
            cin >> dataAddr;
        }
        if (dataAddr == -1)
            break;
        if (dataAddr != -1)
        {
            cout << "Enter Data." << endl;
            cin >> data;
            memory[dataAddr] = data;
        }
        cout << "Enter address of data. Enter -1 if you're done" << endl;
        cin >> dataAddr;
    }
}

void fetch()
{
    if (instrBuffer.size() == 0)
    {
        parsedInst p;
        int i = 0;
        while (i < 4 && pc <= instEnd)
        {
            p = parser(memory[pc], pc);
            p.index = instIndex;
            instrBuffer.push(p);
            pc++;
            i++;

            if ((p.op == "BEQ" && branchPredictor(BITPRED, 1, p.addr, 0, p.imm)) || p.op == "JMP")
            {
                pc += p.imm;
            }

            ClockTracing c;
            c.par = p;
            c.Executed = -1;
            c.count = -1;
            c.fetch = clk;
            c.Issued = -1;
            c.Written = -1;
            c.commit = -1;
            c.flushed = 0;
            trace.push_back(c);
            instIndex++;
        }
    }
}

void issueCall()
{
    if (instrBuffer.size() > 0) //check if instruction buffer is not empty for issue
    {
        parsedInst ali = instrBuffer.front();

        for (int j = 0; j < trace.size(); j++)
        {
            if ((ali.index == trace[j].par.index && trace[j].fetch < clk && trace[j].Issued == -1 && !trace[j].flushed))
            {
                if (ali.op == "ADDI" || ali.op == "ADD" || ali.op == "SUB")
                {
                    issue(addStation, 3, ali, j);
                }
                else if (ali.op == "NAND")
                {
                    issue(nandStation, 1, ali, j);
                }
                else if (ali.op == "MUL")
                {
                    issue(multStation, 2, ali, j);
                }
                else if (ali.op == "BEQ")
                {
                    issue(beqStation, 2, ali, j);
                }
                else if (ali.op == "JMP" || ali.op == "JALR" || ali.op == "RET")
                {
                    issue(jumpStation, 3, ali, j);
                }
                else if (ali.op == "LW")
                {
                    issue(loadStation, 2, ali, j);
                }
                else if (ali.op == "SW")
                {
                    issue(storeStation, 2, ali, j);
                }
            }
        }
    }

    //BRANCH PREDICTION TO CHECK FOR SECONDISSUE

    if (instrBuffer.size() > 0) //check if instruction buffer is not empty for second issue
    {
        parsedInst aya = instrBuffer.front();
        for (int j = 0; j < trace.size(); j++)
        {
            if (aya.index == trace[j].par.index && trace[j].fetch < clk && trace[j].Issued == -1 && !trace[j].flushed)
            {
                if (aya.op == "ADDI" || aya.op == "ADD" || aya.op == "SUB")
                {
                    issue(addStation, 3, aya, j);
                }
                else if (aya.op == "NAND")
                {
                    issue(nandStation, 1, aya, j);
                }
                else if (aya.op == "MUL")
                {
                    issue(multStation, 2, aya, j);
                }
                else if (aya.op == "BEQ")
                {
                    issue(beqStation, 2, aya, j);
                }
                else if (aya.op == "JMP" || aya.op == "JALR" || aya.op == "RET")
                {
                    issue(jumpStation, 3, aya, j);
                }
                else if (aya.op == "LW")
                {
                    issue(loadStation, 2, aya, j);
                }
                else if (aya.op == "SW")
                {
                    issue(storeStation, 2, aya, j);
                }
            }
        }
    }
}

void issue(reservationStationEntry R[], int size, parsedInst I, int traceIndex)
{
    for (int i = 0; i < size; i++)
    {
        if (R[i].busy == 0)
        {
            if ((head == tail && !ROB[tail].busy) || (!ROB[(tail + 1) % 6].busy))
            {
                if (((head == tail) && ROB[head].busy == 1) || ((head != tail))) // && (!ROB[(tail + 1) % 6].busy))) //TODO: Check second condition
                {
                    tail = (tail + 1) % 6;
                }

                if (I.op == "ADD" || I.op == "SUB" || I.op == "NAND" || I.op == "MUL")
                {
                    if (registerStatus[I.rs1] == -1)
                    {
                        R[i].Vj = registerFile[I.rs1];
                        R[i].Qj = -1;
                    }
                    else
                    {
                        if (ROB[registerStatus[I.rs1]].ready == 1)
                        {
                            R[i].Vj = ROB[registerStatus[I.rs1]].value;
                            R[i].Qj = -1;
                        }
                        else
                        {
                            R[i].Vj = -1;
                            R[i].Qj = registerStatus[I.rs1];
                        }
                    }

                    if (registerStatus[I.rs2] == -1)
                    {
                        R[i].Vk = registerFile[I.rs2];
                        R[i].Qk = -1;
                    }
                    else
                    {
                        if (ROB[registerStatus[I.rs2]].ready == 1)
                        {
                            R[i].Vk = ROB[registerStatus[I.rs2]].value;
                            R[i].Qk = -1;
                        }
                        else
                        {
                            R[i].Vk = -1;
                            R[i].Qk = registerStatus[I.rs2];
                        }
                    }

                    R[i].dest = tail;
                    R[i].addr = -1;
                }
                else if (I.op == "ADDI")
                {
                    if (registerStatus[I.rs1] == -1)
                    {
                        R[i].Vj = registerFile[I.rs1];
                        R[i].Qj = -1;
                    }
                    else
                    {
                        if (ROB[registerStatus[I.rs1]].ready == 1)
                        {
                            R[i].Vj = ROB[registerStatus[I.rs1]].value;
                            R[i].Qj = -1;
                        }
                        else
                        {
                            R[i].Vj = -1;
                            R[i].Qj = registerStatus[I.rs1];
                        }
                    }

                    R[i].Vk = I.imm;
                    R[i].Qk = -1;

                    R[i].dest = tail;
                    R[i].addr = -1;
                }
                else if (I.op == "LW")
                {
                    if (registerStatus[I.rs1] == -1)
                    {
                        R[i].Vj = registerFile[I.rs1];
                        R[i].Qj = -1;
                    }
                    else
                    {
                        if (ROB[registerStatus[I.rs1]].ready == 1)
                        {
                            R[i].Vj = ROB[registerStatus[I.rs1]].value;
                            R[i].Qj = -1;
                        }
                        else
                        {
                            R[i].Vj = -1;
                            R[i].Qj = registerStatus[I.rs1];
                        }
                    }

                    R[i].Vk = -1;
                    R[i].Qk = -1;

                    R[i].dest = tail;
                    R[i].addr = I.imm;
                }
                else if (I.op == "BEQ" || I.op == "SW")
                {
                    if (registerStatus[I.rs1] == -1)
                    {
                        R[i].Vj = registerFile[I.rs1];
                        R[i].Qj = -1;
                    }
                    else
                    {
                        if (ROB[registerStatus[I.rs1]].ready == 1)
                        {
                            R[i].Vj = ROB[registerStatus[I.rs1]].value;
                            R[i].Qj = -1;
                        }
                        else
                        {
                            R[i].Vj = -1;
                            R[i].Qj = registerStatus[I.rs1];
                        }
                    }

                    if (registerStatus[I.rs2] == -1)
                    {
                        R[i].Vk = registerFile[I.rs2];
                        R[i].Qk = -1;
                    }
                    else
                    {
                        if (ROB[registerStatus[I.rs2]].ready == 1)
                        {
                            R[i].Vk = ROB[registerStatus[I.rs2]].value;
                            R[i].Qk = -1;
                        }
                        else
                        {
                            R[i].Vk = -1;
                            R[i].Qk = registerStatus[I.rs2];
                        }
                    }

                    R[i].dest = tail;
                    R[i].addr = I.imm;
                }
                else if (I.op == "JMP")
                {
                    R[i].Vj = -1;
                    R[i].Qj = -1;
                    R[i].Vk = -1;
                    R[i].Qk = -1;

                    R[i].dest = tail;
                    R[i].addr = I.imm;
                }
                else if (I.op == "JALR" || I.op == "RET")
                {
                    if (registerStatus[I.rs1] == -1)
                    {
                        R[i].Vj = registerFile[I.rs1];
                        R[i].Qj = -1;
                    }
                    else
                    {
                        if (ROB[registerStatus[I.rs1]].ready == 1)
                        {
                            R[i].Vj = ROB[registerStatus[I.rs1]].value;
                            R[i].Qj = -1;
                        }
                        else
                        {
                            R[i].Vj = -1;
                            R[i].Qj = registerStatus[I.rs1];
                        }
                    }

                    R[i].Vk = -1;
                    R[i].Qk = -1;
                    R[i].dest = tail;
                    R[i].addr = -1;
                }

                R[i].busy = true;
                R[i].op = I.op;
                R[i].instAddr = I.addr;
                R[i].instIndex = I.index;

                ROB[tail].busy = 1;
                ROB[tail].ready = 0;
                ROB[tail].dest = I.rd;
                ROB[tail].inst = I;

                trace[traceIndex].Issued = clk;

                if (I.rd != 0 && I.rd != -1)
                    registerStatus[I.rd] = tail;

                instrBuffer.pop();
                break;
            }
        }
    }
}

void execution()
{
    //////////////////////////////////////2 Reservation Stations ////////////////////////////////
    for (int i = 0; i < 2; i++)
    {
        // Load
        if ((loadStation[i].Qj == -1) && (loadStation[i].busy == 1)) // load
        {
            int index = 0;
            for (int j = 0; j < trace.size(); j++)
            {
                if (loadStation[i].instIndex == trace[j].par.index)
                {
                    index = j;
                }
            }
            if ((trace[index].Issued < clk) && (trace[index].Issued != -1) && !trace[index].flushed)
            {
                if ((trace[index].Executed == -1) && (trace[index].count == -1))
                {
                    trace[index].count = 2;
                }
                if (trace[index].count == 2)
                {
                    bool flag = true;
                    for (int k = 0; k < 2; k++) // loop on sw functional units
                    {
                        int storeIndex = 0;
                        for (int j = 0; j < trace.size(); j++)
                        {
                            if (loadStation[i].instIndex == trace[j].par.index)
                            {
                                storeIndex = j;
                            }
                        }
                        if (storeStation[k].busy == 1 && (trace[storeIndex].Issued < trace[index].Issued || (trace[storeIndex].Issued == trace[index].Issued && storeStation[k].instAddr < loadStation[i].instAddr))) //check if FU is busy and occupying instruction is before current load
                        {
                            flag = false;
                        }
                    }
                    if (flag) // execute step 1 and move to step 2 of exectution
                    {
                        trace[index].count = 1;
                        loadStation[i].addr += loadStation[i].Vj; // compute load address
                    }
                }
                else if (trace[index].count == 1)
                {
                    bool flag = true;
                    for (int k = 0; k < 6; k++) // loop on ROB
                    {
                        int storeIndex = 0;
                        for (int j = 0; j < trace.size(); j++)
                        {
                            if (loadStation[i].instIndex == trace[j].par.index)
                            {
                                storeIndex = j;
                            }
                        }
                        if (ROB[k].pcAddr == loadStation[i].addr && ROB[k].inst.op == "SW" && ROB[k].busy == 1 && (trace[storeIndex].Issued < trace[index].Issued || (trace[storeIndex].Issued == trace[index].Issued && storeStation[k].instAddr < loadStation[i].instAddr))) //check if ROB contains dependable instructions
                        {
                            flag = false;
                        }
                    }
                    if (flag)
                    {
                        trace[index].count = 0;
                        loadStation[i].Vk = stoi(memory[loadStation[i].addr]); //load value in operand 2
                    }
                }

                if ((trace[index].count == 0) && (trace[index].Executed == -1))
                {
                    trace[index].Executed = clk;
                    trace[index].par.result = loadStation[i].Vk; //assign result in trace the second operand assigned above to the value gotten from memory
                }
            }
        }
        // Store
        if ((storeStation[i].Qj == -1) && (storeStation[i].busy == 1) && (storeStation[i].Qk == -1)) //store
        {

            int index = 0;
            for (int j = 0; j < trace.size(); j++)
            {
                if (storeStation[i].instIndex == trace[j].par.index)
                {
                    index = j;
                }
            }
            if ((trace[index].Issued < clk) && (trace[index].Issued != -1) && !trace[index].flushed)
            {
                if ((trace[index].Executed == -1) && (trace[index].count == -1))
                {
                    trace[index].count = 0;
                }
                else if (trace[index].count != 0)
                {
                    trace[index].count--;
                }
                if ((trace[index].count == 0) && (trace[index].Executed == -1))
                {
                    int n;
                    trace[index].Executed = clk;
                    storeStation[i].addr += storeStation[i].Vk;
                    trace[index].par.pcAddr = storeStation[i].addr;
                    trace[index].par.result = storeStation[i].Vj;
                }
            }
        }
        //BEQ
        if ((beqStation[i].Qj == -1) && (beqStation[i].busy == 1) && (beqStation[i].Qk == -1)) //BEQ
        {

            int index = 0;
            for (int j = 0; j < trace.size(); j++)
            {
                if (beqStation[i].instIndex == trace[j].par.index)
                {
                    index = j;
                }
            }
            if ((trace[index].Issued < clk) && (trace[index].Issued != -1) && !trace[index].flushed)
            {
                if ((trace[index].Executed == -1) && (trace[index].count == -1))
                {
                    trace[index].count = 0;
                }
                else if (trace[index].count != 0)
                {
                    trace[index].count--;
                }
                if ((trace[index].count == 0) && (trace[index].Executed == -1))
                {
                    trace[index].Executed = clk;
                    int v1, v2;
                    if (trace[index].par.index == beqStation[i].instIndex)
                    {
                        v1 = beqStation[i].Vj;
                        v2 = beqStation[i].Vk;
                        if (v1 == v2)
                        {
                            trace[index].par.result = 1;
                            trace[index].par.pcAddr = beqStation[i].instAddr + 1 + beqStation[i].addr;
                        }
                        else
                        {
                            trace[index].par.result = 0;
                            trace[index].par.pcAddr = beqStation[i].instAddr + 1;
                        }
                    }
                }
            }
        }
        //MULT
        if ((multStation[i].Qj == -1) && (multStation[i].busy == 1) && (multStation[i].Qk == -1)) //Mult
        {
            int index = 0;
            for (int j = 0; j < trace.size(); j++)
            {
                if (multStation[i].instIndex == trace[j].par.index)
                {
                    index = j;
                }
            }
            if ((trace[index].Issued < clk) && (trace[index].Issued != -1) && !trace[index].flushed)
            {
                if ((trace[index].Executed == -1) && (trace[index].count == -1))
                {
                    trace[index].count = 7;
                }
                else if (trace[index].count != 0)
                {
                    trace[index].count--;
                }
                if ((trace[index].count == 0) && (trace[index].Executed == -1))
                {
                    int v1, v2;
                    v1 = multStation[i].Vj;
                    v2 = multStation[i].Vk;
                    trace[index].par.result = v1 * v2;
                    trace[index].Executed = clk;
                }
            }
        }
    }
    /////////////////////////3  Reservation Stations////////////////////////////////////////////////
    for (int i = 0; i < 3; i++)
    {
        if ((addStation[i].Qj == -1) && (addStation[i].busy == 1) && (addStation[i].Qk == -1)) //Add,Sub,Addi
        {
            int index = 0;
            for (int j = 0; j < trace.size(); j++)
            {
                if (addStation[i].instIndex == trace[j].par.index)
                {
                    index = j;
                }
            }

            if ((trace[index].Issued < clk) && (trace[index].Issued != -1) && !trace[index].flushed)
            {
                if ((trace[index].Executed == -1) && (trace[index].Issued != -1) && (trace[index].count == -1))
                {
                    trace[index].count = 1;
                }
                else if (trace[index].count != 0)
                {
                    trace[index].count--;
                }
                if ((trace[index].count == 0) && (trace[index].Executed == -1))
                {
                    trace[index].Executed = clk;
                    if (trace[index].par.op == "ADD")
                    {
                        int v1, v2;
                        v1 = addStation[i].Vj;
                        v2 = addStation[i].Vk;
                        trace[index].par.result = v1 + v2;
                    }
                    else if (trace[index].par.op == "SUB")
                    {
                        int v1, v2;
                        v1 = addStation[i].Vj;
                        v2 = addStation[i].Vk;
                        trace[index].par.result = v1 - v2;
                    }
                    if (trace[index].par.op == "ADDI")
                    {
                        int v1, v2;
                        v1 = addStation[i].Vj;
                        v2 = addStation[i].Vk;
                        trace[index].par.result = v1 + v2;
                    }
                }
            }
        }
        //JALR, RET
        if ((jumpStation[i].Qj == -1) && (jumpStation[i].busy == 1) && (jumpStation[i].op != "JMP")) //Jalr,Ret
        {
            int index = 0;
            for (int j = 0; j < trace.size(); j++)
            {
                if (jumpStation[i].instIndex == trace[j].par.index)
                {
                    index = j;
                }
            }
            if ((trace[index].Issued < clk) && (trace[index].Issued != -1) && !trace[index].flushed)
            {
                if ((trace[index].Executed == -1) && (trace[index].count == -1))
                {
                    trace[index].count = 0;
                }
                else if (trace[index].count != 0)
                {
                    trace[index].count--;
                }
                if ((trace[index].count == 0) && (trace[index].Executed == -1))
                {
                    trace[index].Executed = clk;
                    if (trace[index].par.addr == jumpStation[i].instAddr)
                    {
                        if (trace[index].par.op == "JALR")
                        {
                            int v1;
                            v1 = jumpStation[i].Vj;
                            trace[index].par.result = trace[index].par.addr + 1;
                            trace[index].par.pcAddr = v1;
                        }
                        else if (trace[index].par.op == "RET")
                        {
                            int v1;
                            v1 = jumpStation[i].Vj;
                            trace[index].par.pcAddr = v1;
                        }
                    }
                }
            }
        }
        //JMP
        if ((jumpStation[i].busy == 1) && (jumpStation[i].op == "JMP")) //Jmp
        {
            int index = 0;
            for (int j = 0; j < trace.size(); j++)
            {
                if (jumpStation[i].instIndex == trace[j].par.index)
                {
                    index = j;
                }
            }
            if ((trace[index].Issued < clk) && (trace[index].Issued != -1) && !trace[index].flushed)
            {
                if ((trace[index].Executed == -1) && (trace[index].Issued != -1) && (trace[index].count == -1))
                {
                    trace[index].count = 0;
                }
                else if (trace[index].count != 0)
                {
                    trace[index].count--;
                }
                if ((trace[index].count == 0) && (trace[index].Executed == -1))
                {
                    trace[index].Executed = clk;
                    int v1;
                    v1 = jumpStation[i].addr;
                    trace[index].par.pcAddr = v1;
                }
            }
        }
    }

    if ((nandStation[0].Qj == -1) && (nandStation[0].busy == 1) && (nandStation[0].Qk == -1)) //nand
    {
        int index = 0;
        for (int j = 0; j < trace.size(); j++)
        {
            if (nandStation[0].instIndex == trace[j].par.index)
            {
                index = j;
            }
        }
        if ((trace[index].Issued < clk) && (trace[index].Issued != -1) && !trace[index].flushed)
        {
            if ((trace[index].Executed == -1) && (trace[index].Issued != -1) && (trace[index].count == -1))
            {
                trace[index].count = 0;
            }
            else if (trace[index].count != 0)
            {
                trace[index].count--;
            }
            if ((trace[index].count == 0) && (trace[index].Executed == -1))
            {
                trace[index].Executed = clk;
                int v1, v2;
                v1 = nandStation[0].Vj;
                v2 = nandStation[0].Vk;
                trace[index].par.result = !(v1 & v2);
            }
        }
    }
}

void write()
{
    int writes = 0;
    for (int i = 0; i < trace.size(); i++)
    {
        if (trace[i].Executed < clk && trace[i].Written == -1 && trace[i].Executed != -1 && writes < 2 && !trace[i].flushed)
        {
            trace[i].Written = clk;

            writes++;

            int index = trace[i].par.index;

            int d;

            for (int j = 0; j < 6; j++)
            {
                if (ROB[j].inst.index == trace[i].par.index && ROB[j].busy == 1)
                    d = j;
            }

            ROB[d].value = trace[i].par.result;
            ROB[d].pcAddr = trace[i].par.pcAddr;
            ROB[d].ready = 1;

            if (trace[i].par.op == "ADD" || trace[i].par.op == "SUB" || trace[i].par.op == "ADDI")
            {
                for (int j = 0; j < 3; j++)
                {
                    if (addStation[j].instIndex == index)
                    {
                        addStation[j].busy = 0;
                        addStation[j].op = "";
                        addStation[j].Vj = -1;
                        addStation[j].Vk = -1;
                        addStation[j].Qj = -1;
                        addStation[j].Qk = -1;
                        addStation[j].dest = -1;
                        addStation[j].addr = -1;
                        addStation[j].instAddr = -1;
                        addStation[j].instIndex = -1;
                    }
                }
            }
            else if (trace[i].par.op == "JALR" || trace[i].par.op == "JMP" || trace[i].par.op == "RET")
            {
                for (int j = 0; j < 3; j++)
                {
                    if (jumpStation[j].instIndex == index)
                    {
                        jumpStation[j].busy = 0;
                        jumpStation[j].op = "";
                        jumpStation[j].Vj = -1;
                        jumpStation[j].Vk = -1;
                        jumpStation[j].Qj = -1;
                        jumpStation[j].Qk = -1;
                        jumpStation[j].dest = -1;
                        jumpStation[j].addr = -1;
                        jumpStation[j].instAddr = -1;
                        jumpStation[j].instIndex = -1;
                    }
                }
            }
            else if (trace[i].par.op == "LW")
            {
                for (int j = 0; j < 2; j++)
                {
                    if (loadStation[j].instIndex == index)
                    {
                        loadStation[j].busy = 0;
                        loadStation[j].op = "";
                        loadStation[j].Vj = -1;
                        loadStation[j].Vk = -1;
                        loadStation[j].Qj = -1;
                        loadStation[j].Qk = -1;
                        loadStation[j].dest = -1;
                        loadStation[j].addr = -1;
                        loadStation[j].instAddr = -1;
                        loadStation[j].instIndex = -1;
                    }
                }
            }
            else if (trace[i].par.op == "SW")
            {
                for (int j = 0; j < 2; j++)
                {
                    if (storeStation[j].instIndex == index)
                    {
                        storeStation[j].busy = 0;
                        storeStation[j].op = "";
                        storeStation[j].Vj = -1;
                        storeStation[j].Vk = -1;
                        storeStation[j].Qj = -1;
                        storeStation[j].Qk = -1;
                        storeStation[j].dest = -1;
                        storeStation[j].addr = -1;
                        storeStation[j].instAddr = -1;
                        storeStation[j].instIndex = -1;
                    }
                }
            }
            else if (trace[i].par.op == "BEQ")
            {
                for (int j = 0; j < 2; j++)
                {
                    if (beqStation[j].instIndex == index)
                    {
                        beqStation[j].busy = 0;
                        beqStation[j].op = "";
                        beqStation[j].Vj = -1;
                        beqStation[j].Vk = -1;
                        beqStation[j].Qj = -1;
                        beqStation[j].Qk = -1;
                        beqStation[j].dest = -1;
                        beqStation[j].addr = -1;
                        beqStation[j].instAddr = -1;
                        beqStation[j].instIndex = -1;
                    }
                }
            }
            else if (trace[i].par.op == "MUL")
            {
                for (int j = 0; j < 2; j++)
                {
                    if (multStation[j].instIndex == index)
                    {
                        multStation[j].busy = 0;
                        multStation[j].op = "";
                        multStation[j].Vj = -1;
                        multStation[j].Vk = -1;
                        multStation[j].Qj = -1;
                        multStation[j].Qk = -1;
                        multStation[j].dest = -1;
                        multStation[j].addr = -1;
                        multStation[j].instAddr = -1;
                        multStation[j].instIndex = -1;
                    }
                }
            }
            else if (trace[i].par.op == "NAND")
            {
                nandStation[0].busy = 0;
                nandStation[0].op = "";
                nandStation[0].Vj = -1;
                nandStation[0].Vk = -1;
                nandStation[0].Qj = -1;
                nandStation[0].Qk = -1;
                nandStation[0].dest = -1;
                nandStation[0].addr = -1;
                nandStation[0].instAddr = -1;
                nandStation[0].instIndex = -1;
            }

            for (int i = 0; i < 3; i++) //jmp/jalr/ret unit & add/sub/addi unit
            {
                if (jumpStation[i].Qj == d)
                {
                    ///or input value!!!!!!
                    jumpStation[i].Vj = ROB[d].value;
                    jumpStation[i].Qj = -1;
                }
                if (jumpStation[i].Qk == d)
                {
                    ///or input value!!!!!!
                    jumpStation[i].Vk = ROB[d].value;
                    jumpStation[i].Qk = -1;
                }
                if (addStation[i].Qj == d)
                {
                    ///or input value!!!!!!
                    addStation[i].Vj = ROB[d].value;
                    addStation[i].Qj = -1;
                }
                if (addStation[i].Qk == d)
                {
                    ///or input value!!!!!!
                    addStation[i].Vk = ROB[d].value;
                    addStation[i].Qk = -1;
                }
            }

            for (int i = 0; i < 2; i++) //lw unit, sw unit, beq unit, mult
            {
                if (loadStation[i].Qj == d)
                {
                    ///or input value!!!!!!
                    loadStation[i].Vj = ROB[d].value;
                    loadStation[i].Qj = -1;
                }
                if (loadStation[i].Qk == d)
                {
                    ///or input value!!!!!!
                    loadStation[i].Vk = ROB[d].value;
                    loadStation[i].Qk = -1;
                }
                if (storeStation[i].Qj == d)
                {
                    ///or input value!!!!!!
                    storeStation[i].Vj = ROB[d].value;
                    storeStation[i].Qj = -1;
                }
                if (storeStation[i].Qk == d)
                {
                    ///or input value!!!!!!
                    storeStation[i].Vk = ROB[d].value;
                    storeStation[i].Qk = -1;
                }
                if (beqStation[i].Qj == d)
                {
                    ///or input value!!!!!!
                    beqStation[i].Vj = ROB[d].value;
                    beqStation[i].Qj = -1;
                }
                if (beqStation[i].Qk == d)
                {
                    ///or input value!!!!!!
                    beqStation[i].Vk = ROB[d].value;
                    beqStation[i].Qk = -1;
                }
                if (multStation[i].Qj == d)
                {
                    ///or input value!!!!!!
                    multStation[i].Vj = ROB[d].value;
                    multStation[i].Qj = -1;
                }
                if (multStation[i].Qk == d)
                {
                    ///or input value!!!!!!
                    multStation[i].Vk = ROB[d].value;
                    multStation[i].Qk = -1;
                }
            }

            //nand unit
            if (nandStation[0].Qj == d)
            {
                ///or input value!!!!!!
                nandStation[0].Vj = ROB[d].value;
                nandStation[0].Qj = -1;
            }
            if (nandStation[0].Qk == d)
            {
                ///or input value!!!!!!
                nandStation[0].Vk = ROB[d].value;
                nandStation[0].Qk = -1;
            }
        }
    }
}

void commit()
{
    int writes = 0;

    for (int i = 0; i < trace.size(); i++)
    { //find the instruction of the head in trace
        if (trace[i].Written < clk && trace[i].commit == -1 && trace[i].Written != -1 && writes < 2 && !trace[i].flushed)
        {
            if (ROB[head].ready)
            { //if ROB head is ready
                if (ROB[head].inst.index == trace[i].par.index)
                {
                    bool flushFlag = false;
                    if (ROB[head].inst.op == "BEQ") //if this is a branch instruction
                    {
                        flushFlag = branchPredictor(BITPRED, 0, trace[i].par.addr, ROB[head].value, trace[i].par.imm);
                        countBranch++;
                    }
                    else if (ROB[head].inst.op == "SW") //if this is a store instruction
                    {
                        memory[ROB[head].pcAddr] = to_string(ROB[head].value);
                    }
                    //if this is a JALR instruction
                    else if (ROB[head].inst.op == "JALR")
                    {
                        registerFile[ROB[head].dest] = ROB[head].value;
                        flushFlag = true;
                    }
                    //if this is a RET instruction
                    else if (ROB[head].inst.op == "RET")
                    {
                        flushFlag = true;
                    }
                    else if (ROB[head].inst.op == "JMP")
                    {
                    }
                    else //if this is another instruction
                    {
                        registerFile[ROB[head].dest] = ROB[head].value;
                    }
                    //store the clock cycle in whitch we commited
                    trace[i].commit = clk;
                    writes++;
                    countCommit++;
                    //set ROB head as not busy and increment
                    ROB[head].busy = false;

                    if (registerStatus[ROB[head].dest] == head)
                    {
                        registerStatus[ROB[head].dest] = -1;
                    }

                    if (flushFlag)
                    {
                        pc = ROB[head].pcAddr;
                        for (int k = i + 1; k < trace.size(); k++)
                        {
                            trace[k].flushed = 1;
                        }
                        flush();
                    }

                    if (head != tail)
                    {
                        head = (head + 1) % robSize;
                    }
                }
            }
        }
    }

    if (head == tail)
    {
        if (ROB[head].busy == false)
        {
            done = 1;
        }
        else
        {
            done = 0;
        }
    }
    else if (ROB[head].busy == true)
    {
        done = 0;
    }
}

bool branchPredictor(int bitsPrediction, bool mode, int address, bool Bresult, int imm)
{

    if (bitsPrediction == 2)
    {
        //predict mode
        if (mode)
        {
            //scan the buffer
            for (int i = 0; i < predictionBuffer.size(); i++)
            {
                //if there is an entry for this address
                if (predictionBuffer[i].adr == address)
                {
                    if (predictionBuffer[i].state < 2) //states 0, 1, 2, 3 --> SNT, NT, T, ST
                        return 0;
                    else
                        return 1;
                }
            }
            //if this is the first time this address is encountered
            predictorEntry newEntry;
            newEntry.adr = address;
            newEntry.state = 2; //initial prediction state is assumed to be taken.
            predictionBuffer.push_back(newEntry);
            return 1;
        }
        //update mode (here we update buffer depending on actual branch result)
        if (!mode)
        {
            int location = -1;

            for (int i = 0; i < predictionBuffer.size(); i++)
            {
                //if there is an entry for this address
                if (predictionBuffer[i].adr == address)
                {
                    location = i;
                }
            }
            bool flush;
            if (Bresult && (predictionBuffer[location].state == 2 || predictionBuffer[location].state == 3)) //Prediction correct
                flush = 0;
            if (Bresult && (predictionBuffer[location].state == 0 || predictionBuffer[location].state == 1)) //Prediction incorrect
                flush = 1;
            if (!Bresult && (predictionBuffer[location].state == 0 || predictionBuffer[location].state == 1)) //Prediction correct
                flush = 0;
            if (!Bresult && (predictionBuffer[location].state == 2 || predictionBuffer[location].state == 3)) //Prediction incorrect
                flush = 1;

            //update states
            if (predictionBuffer[location].state == 0 && Bresult) //current state is SNT
                predictionBuffer[location].state = 1;
            else if (predictionBuffer[location].state == 1)
            { //current state is NT

                if (Bresult)
                    predictionBuffer[location].state = 2;
                else
                    predictionBuffer[location].state = 0;
            }
            else if (predictionBuffer[location].state == 2)
            { //current state is T

                if (Bresult)
                    predictionBuffer[location].state = 3;
                else
                    predictionBuffer[location].state = 1;
            }
            else if (predictionBuffer[location].state == 3 && (!Bresult)) //current state is ST
                predictionBuffer[location].state = 2;

            if (flush == 1)
                countMissBranch++;
            return flush;
        }
    }
    else if (bitsPrediction == 1)
    {
        //predict mode
        if (mode)
        {
            //scan the buffer
            for (int i = 0; i < predictionBuffer.size(); i++)
            {
                //if there is an entry for this address
                if (predictionBuffer[i].adr == address)
                {
                    if (predictionBuffer[i].state == 0) //states 0, 1, --> NT, T
                        return 0;
                    else
                        return 1;
                }
            }
            //if this is the first time this address is encountered
            predictorEntry newEntry;
            newEntry.adr = address;
            newEntry.state = 1; //initial prediction state is assumed to be taken.
            predictionBuffer.push_back(newEntry);
            return 1;
        }
        //update mode (here we update buffer depending on actual branch result)
        if (!mode)
        {
            int location = -1;

            for (int i = 0; i < predictionBuffer.size(); i++)
            {
                //if there is an entry for this address
                if (predictionBuffer[i].adr == address)
                {
                    location = i;
                }
            }
            bool flush;
            if (predictionBuffer[location].state == Bresult) //no flush
                flush = 0;
            else
            { //flush
                countMissBranch++;
                flush = 1;
            }
            //update states
            predictionBuffer[location].state = Bresult;

            return flush;
        }
    }
    else if (bitsPrediction == 0) //static prediction
    {
        if (mode) //prediction mode
        {
            for (int i = 0; i < predictionBuffer.size(); i++)
            {
                //if there is an entry for this address
                if (predictionBuffer[i].adr == address)
                {
                    if (imm < 0) //taken if offset is negative
                    {
                        predictionBuffer[i].state = 1;
                        return 1;
                    }
                    else //not taken otherwise
                    {
                        predictionBuffer[i].state = 0;
                        return 0;
                    }
                }
            }
            //new entry
            predictorEntry newEntry;
            newEntry.adr = address;
            if (imm < 0) //taken if offset is negative
            {
                newEntry.state = 1;
                predictionBuffer.push_back(newEntry);
                return 1;
            }
            else //not taken otherwise
            {
                newEntry.state = 0;
                predictionBuffer.push_back(newEntry);
                return 0;
            }
        }
        else //update mode
        {
            for (int i = 0; i < predictionBuffer.size(); i++)
            {
                //if there is an entry for this address
                if (predictionBuffer[i].adr == address)
                {
                    if (predictionBuffer[i].state == Bresult)
                    {             //if was correct
                        return 0; //dont flush
                    }
                    if (predictionBuffer[i].state != Bresult)
                    {
                        countMissBranch++;
                        return 1; //flush
                    }
                }
            }
        }
    }
    return 0;
}

void InitializeRegisters()
{
    for (int i = 0; i < 8; i++)
        registerStatus[i] = -1;
}

void resetROB()
{
    for (int i = 0; i < 6; i++)
    {
        ROB[i].busy = 0;
        ROB[i].dest = -1;
        ROB[i].ready = 0;
        ROB[i].inst.addr = -1;
    }
    head = 0;
    tail = 0;
}

void resetInstrBuffer()
{
    while (instrBuffer.size() > 0)
    {
        parsedInst p;
        p = instrBuffer.front();
        instrBuffer.pop();
    }
}

void resetReservation()
{
    //reset add stations
    for (int j = 0; j < 3; j++)
    {
        addStation[j].busy = 0;
        addStation[j].op = "";
        addStation[j].Vj = -1;
        addStation[j].Vk = -1;
        addStation[j].Qj = -1;
        addStation[j].Qk = -1;
        addStation[j].dest = -1;
        addStation[j].addr = -1;
        addStation[j].instAddr = -1;
    }
    //reset jump stations
    for (int j = 0; j < 3; j++)
    {
        jumpStation[j].busy = 0;
        jumpStation[j].op = "";
        jumpStation[j].Vj = -1;
        jumpStation[j].Vk = -1;
        jumpStation[j].Qj = -1;
        jumpStation[j].Qk = -1;
        jumpStation[j].dest = -1;
        jumpStation[j].addr = -1;
        jumpStation[j].instAddr = -1;
    }
    //reset load stations
    for (int j = 0; j < 2; j++)
    {
        loadStation[j].busy = 0;
        loadStation[j].op = "";
        loadStation[j].Vj = -1;
        loadStation[j].Vk = -1;
        loadStation[j].Qj = -1;
        loadStation[j].Qk = -1;
        loadStation[j].dest = -1;
        loadStation[j].addr = -1;
        loadStation[j].instAddr = -1;
    }
    //reset store stations
    for (int j = 0; j < 2; j++)
    {
        storeStation[j].busy = 0;
        storeStation[j].op = "";
        storeStation[j].Vj = -1;
        storeStation[j].Vk = -1;
        storeStation[j].Qj = -1;
        storeStation[j].Qk = -1;
        storeStation[j].dest = -1;
        storeStation[j].addr = -1;
        storeStation[j].instAddr = -1;
    }
    //reset branch stations
    for (int j = 0; j < 2; j++)
    {
        beqStation[j].busy = 0;
        beqStation[j].op = "";
        beqStation[j].Vj = -1;
        beqStation[j].Vk = -1;
        beqStation[j].Qj = -1;
        beqStation[j].Qk = -1;
        beqStation[j].dest = -1;
        beqStation[j].addr = -1;
        beqStation[j].instAddr = -1;
    }
    //reset mult stations
    for (int j = 0; j < 2; j++)
    {
        multStation[j].busy = 0;
        multStation[j].op = "";
        multStation[j].Vj = -1;
        multStation[j].Vk = -1;
        multStation[j].Qj = -1;
        multStation[j].Qk = -1;
        multStation[j].dest = -1;
        multStation[j].addr = -1;
        multStation[j].instAddr = -1;
    }
    //reset nand stations
    nandStation[0].busy = 0;
    nandStation[0].op = "";
    nandStation[0].Vj = -1;
    nandStation[0].Vk = -1;
    nandStation[0].Qj = -1;
    nandStation[0].Qk = -1;
    nandStation[0].dest = -1;
    nandStation[0].addr = -1;
    nandStation[0].instAddr = -1;
}

void flush() //when flushing because of branch/JMP/JALR/RET (assuming they take place in commit
{
    resetReservation();
    resetROB();
    resetInstrBuffer();
    resetRAT();
}

void resetRAT()
{
    for (int i = 0; i < 8; i++)
    {
        registerStatus[i] = -1;
    }
}

void outputTable()
{
    cout << endl;
    cout << setw(20) << left << "Instruction"
         << setw(10) << left << "FLUSHED"
         << setw(10) << left << "FETCHED"
         << setw(10) << left << "ISSUED"
         << setw(10) << left << "EXECUTED"
         << setw(10) << left << "WRITTEN"
         << setw(10) << left << "COMMITTED" << endl;
    cout << "--------------------------------------------------------------------------------" << endl;
    for (int i = 0; i < trace.size(); i++)
    {
        cout << setw(20) << left << trace[i].par.inst
             << setw(10) << left << trace[i].flushed
             << setw(10) << left << trace[i].fetch
             << setw(10) << left << trace[i].Issued
             << setw(10) << left << trace[i].Executed
             << setw(10) << left << trace[i].Written
             << setw(10) << left << trace[i].commit << endl;
    }
    //    cout << endl;
}

void outputValues()
{
    cout << "Number of commitedd instructions is: " << countCommit << endl;
    int last = trace.size() - 1;
    while (trace[last].commit == -1)
        last--;
    cout << "Number of cycles: " << trace[last].commit << endl;
    cout << "IPC = " << double(countCommit) / trace[last].commit << endl;
    cout << "Total Branch Count = " << double(countBranch) << endl;
    cout << "Branches Miss-Predicted = " << double(countMissBranch) << endl;
    cout << "Branch Miss Prediction Rate = " << 100.0 * (double(countMissBranch) / double(countBranch)) << "%" << endl;
}
