/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE464_Insertion_Of_Data_Structure_Sentinel__basic_68a.c
Label Definition File: CWE464_Insertion_Of_Data_Structure_Sentinel__basic.label.xml
Template File: sources-sink-68a.tmpl.c
*/
/*
 * @description
 * CWE: 464 Insertion of Data Structure Sentinel
 * BadSource:  Read in data from the console and convert to an int
 * GoodSource: Set data to a fixed char
 * Sink:
 *    BadSink : Place data into and print an array
 * Flow Variant: 68 Data flow: data passed as a global variable from one function to another in different source files
 *
 * */

#include "std_testcase.h"

char CWE464_Insertion_Of_Data_Structure_Sentinel__basic_68_bad_data;
char CWE464_Insertion_Of_Data_Structure_Sentinel__basic_68_goodG2B_data;

#ifndef OMITBAD

/* bad function declaration */
void CWE464_Insertion_Of_Data_Structure_Sentinel__basic_68b_bad_sink();

void CWE464_Insertion_Of_Data_Structure_Sentinel__basic_68_bad()
{
    char data;
    data = ' ';
    {
        char ch;
        ch = (char)getc(stdin);
        /* FLAW: If the character entered on the command line is not an int,
         * a null value will be returned */
        data = (char)atoi(&ch);
    }
    CWE464_Insertion_Of_Data_Structure_Sentinel__basic_68_bad_data = data;
    CWE464_Insertion_Of_Data_Structure_Sentinel__basic_68b_bad_sink();
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good function declarations */
void CWE464_Insertion_Of_Data_Structure_Sentinel__basic_68b_goodG2B_sink();

/* goodG2B uses the GoodSource with the BadSink */
static void goodG2B()
{
    char data;
    data = ' ';
    /* FIX: Set data to be a char */
    data = 'a';
    CWE464_Insertion_Of_Data_Structure_Sentinel__basic_68_goodG2B_data = data;
    CWE464_Insertion_Of_Data_Structure_Sentinel__basic_68b_goodG2B_sink();
}

void CWE464_Insertion_Of_Data_Structure_Sentinel__basic_68_good()
{
    goodG2B();
}

#endif /* OMITGOOD */

/* Below is the main(). It is only used when building this testcase on
   its own for testing or for building a binary to use in testing binary
   analysis tools. It is not used when compiling all the testcases as one
   application, which is how source code analysis tools are tested. */

#ifdef INCLUDEMAIN

int main(int argc, char * argv[])
{
    /* seed randomness */
    srand( (unsigned)time(NULL) );
#ifndef OMITGOOD
    printLine("Calling good()...");
    CWE464_Insertion_Of_Data_Structure_Sentinel__basic_68_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE464_Insertion_Of_Data_Structure_Sentinel__basic_68_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif