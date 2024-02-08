/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE467_Use_of_sizeof_on_Pointer_Type__double_61a.c
Label Definition File: CWE467_Use_of_sizeof_on_Pointer_Type.label.xml
Template File: sources-sink-61a.tmpl.c
*/
/*
 * @description
 * CWE: 467 Use of sizeof() on a Pointer Type
 * BadSource:  Initialize the source buffer using the size of a pointer
 * GoodSource: Initialize the source buffer using the size of the DataElementType
 * Sinks:
 *    BadSink : Print then free data
 * Flow Variant: 61 Data flow: data returned from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#ifndef OMITBAD

/* bad function declaration */
double * CWE467_Use_of_sizeof_on_Pointer_Type__double_61b_bad_source(double * data);

void CWE467_Use_of_sizeof_on_Pointer_Type__double_61_bad()
{
    double * data;
    /* Initialize data */
    data = NULL;
    data = CWE467_Use_of_sizeof_on_Pointer_Type__double_61b_bad_source(data);
    /* POTENTIAL FLAW: Attempt to use data, which may not have enough memory allocated */
    printDoubleLine(*data);
    free(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
double * CWE467_Use_of_sizeof_on_Pointer_Type__double_61b_goodG2B_source(double * data);

static void goodG2B()
{
    double * data;
    /* Initialize data */
    data = NULL;
    data = CWE467_Use_of_sizeof_on_Pointer_Type__double_61b_goodG2B_source(data);
    /* POTENTIAL FLAW: Attempt to use data, which may not have enough memory allocated */
    printDoubleLine(*data);
    free(data);
}

void CWE467_Use_of_sizeof_on_Pointer_Type__double_61_good()
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
    CWE467_Use_of_sizeof_on_Pointer_Type__double_61_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE467_Use_of_sizeof_on_Pointer_Type__double_61_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
