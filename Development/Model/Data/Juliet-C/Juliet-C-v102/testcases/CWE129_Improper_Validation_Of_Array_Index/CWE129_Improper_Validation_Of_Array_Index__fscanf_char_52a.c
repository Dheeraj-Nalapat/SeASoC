/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE129_Improper_Validation_Of_Array_Index__fscanf_char_52a.c
Label Definition File: CWE129_Improper_Validation_Of_Array_Index.label.xml
Template File: sources-sinks-52a.tmpl.c
*/
/*
 * @description
 * CWE: 129 Improper Validation of Array Index
 * BadSource: fscanf Read data from the console using fscanf()
 * GoodSource: Larger than zero but less than 10
 * Sinks: char
 *    GoodSink: Ensure the array index is valid
 *    BadSink : Improperly check the array index by not checking the upper bound
 * Flow Variant: 52 Data flow: data passed as an argument from one function to another to another in three different source files
 *
 * */

#include "std_testcase.h"

#ifndef OMITBAD

/* bad function declaration */
void CWE129_Improper_Validation_Of_Array_Index__fscanf_char_52b_bad_sink(int data);

void CWE129_Improper_Validation_Of_Array_Index__fscanf_char_52_bad()
{
    int data;
    data = -1; /* Initialize data */
    fscanf (stdin, "%d", &data);
    CWE129_Improper_Validation_Of_Array_Index__fscanf_char_52b_bad_sink(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE129_Improper_Validation_Of_Array_Index__fscanf_char_52b_goodG2B_sink(int data);

static void goodG2B()
{
    int data;
    data = -1; /* Initialize data */
    /* FIX: Use a value greater than 0, but less than 10 to avoid attempting to
     * access an index of the array in the sink that is out-of-bounds */
    data = 7;
    CWE129_Improper_Validation_Of_Array_Index__fscanf_char_52b_goodG2B_sink(data);
}

/* goodB2G uses the BadSource with the GoodSink */
void CWE129_Improper_Validation_Of_Array_Index__fscanf_char_52b_goodB2G_sink(int data);

static void goodB2G()
{
    int data;
    data = -1; /* Initialize data */
    fscanf (stdin, "%d", &data);
    CWE129_Improper_Validation_Of_Array_Index__fscanf_char_52b_goodB2G_sink(data);
}

void CWE129_Improper_Validation_Of_Array_Index__fscanf_char_52_good()
{
    goodG2B();
    goodB2G();
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
    CWE129_Improper_Validation_Of_Array_Index__fscanf_char_52_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE129_Improper_Validation_Of_Array_Index__fscanf_char_52_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
