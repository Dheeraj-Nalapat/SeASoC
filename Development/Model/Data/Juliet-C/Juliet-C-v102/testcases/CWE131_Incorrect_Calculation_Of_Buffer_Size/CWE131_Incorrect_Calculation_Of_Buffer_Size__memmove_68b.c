/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE131_Incorrect_Calculation_Of_Buffer_Size__memmove_68b.c
Label Definition File: CWE131_Incorrect_Calculation_Of_Buffer_Size.missing_sizeof.label.xml
Template File: sources-sink-68b.tmpl.c
*/
/*
 * @description
 * CWE: 131 Incorrect Calculation of Buffer Size
 * BadSource:  Allocate memory without using sizeof(int)
 * GoodSource: Allocate memory using sizeof(int)
 * Sink: memmove
 *    BadSink : Copy array to data using memmove()
 * Flow Variant: 68 Data flow: data passed as a global variable from one function to another in different source files
 *
 * */

#include "std_testcase.h"

extern int * CWE131_Incorrect_Calculation_Of_Buffer_Size__memmove_68_bad_data;
extern int * CWE131_Incorrect_Calculation_Of_Buffer_Size__memmove_68_goodG2B_data;

/* all the sinks are the same, we just want to know where the hit originated if a tool flags one */

#ifndef OMITBAD

void CWE131_Incorrect_Calculation_Of_Buffer_Size__memmove_68b_bad_sink()
{
    int * data = CWE131_Incorrect_Calculation_Of_Buffer_Size__memmove_68_bad_data;
    {
        int data_src[10] = {0};
        /* POTENTIAL FLAW: Possible buffer overflow if data was not allocated correctly in the source */
        memmove(data, data_src, 10*sizeof(int));
        printIntLine(data[0]);
        free(data);
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE131_Incorrect_Calculation_Of_Buffer_Size__memmove_68b_goodG2B_sink()
{
    int * data = CWE131_Incorrect_Calculation_Of_Buffer_Size__memmove_68_goodG2B_data;
    {
        int data_src[10] = {0};
        /* POTENTIAL FLAW: Possible buffer overflow if data was not allocated correctly in the source */
        memmove(data, data_src, 10*sizeof(int));
        printIntLine(data[0]);
        free(data);
    }
}

#endif /* OMITGOOD */