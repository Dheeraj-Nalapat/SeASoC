/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE369_Divide_By_Zero__int_zero_divide_65a.c
Label Definition File: CWE369_Divide_By_Zero__int.label.xml
Template File: sources-sink-65a.tmpl.c
*/
/*
 * @description
 * CWE: 369 Divide By Zero
 * BadSource: zero Fixed value of zero
 * GoodSource: Non-zero
 * Sinks: divide
 *    BadSink : Divide a constant by data
 * Flow Variant: 65 Data/control flow: data passed as an argument from one function to a function in a different source file called via a function pointer
 *
 * */

#include "std_testcase.h"

#ifndef OMITBAD

/* bad function declaration */
void CWE369_Divide_By_Zero__int_zero_divide_65b_bad_sink(int data);

void CWE369_Divide_By_Zero__int_zero_divide_65_bad()
{
    int data;
    /* define a function pointer */
    void (*func_ptr) (int) = CWE369_Divide_By_Zero__int_zero_divide_65b_bad_sink;
    /* Initialize data */
    data = -1;
    /* FLAW: Set data to zero */
    data = 0;
    /* use the function pointer */
    func_ptr(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE369_Divide_By_Zero__int_zero_divide_65b_goodG2B_sink(int data);

static void goodG2B()
{
    int data;
    void (*func_ptr) (int) = CWE369_Divide_By_Zero__int_zero_divide_65b_goodG2B_sink;
    /* Initialize data */
    data = -1;
    /* FIX: Use a value not equal to zero */
    data = 7;
    func_ptr(data);
}

void CWE369_Divide_By_Zero__int_zero_divide_65_good()
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
    CWE369_Divide_By_Zero__int_zero_divide_65_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE369_Divide_By_Zero__int_zero_divide_65_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
