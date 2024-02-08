/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE197_Numeric_Truncation_Error__int_large_to_char_44.c
Label Definition File: CWE197_Numeric_Truncation_Error__int.label.xml
Template File: sources-sink-44.tmpl.c
*/
/*
 * @description
 * CWE: 197 Numeric Truncation Error
 * BadSource: large Set data to a number larger than SHRT_MAX
 * GoodSource: Less than CHAR_MAX
 * Sinks: to_char
 *    BadSink : Convert data to a char
 * Flow Variant: 44 Data/control flow: data passed as an argument from one function to a function in the same source file called via a function pointer
 *
 * */

#include "std_testcase.h"

#ifndef OMITBAD

static void bad_sink(int data)
{
    {
        /* POTENTIAL FLAW: Convert data to a char, possibly causing a truncation error */
        char c = (char)data;
        printHexCharLine(c);
    }
}

void CWE197_Numeric_Truncation_Error__int_large_to_char_44_bad()
{
    int data;
    /* define a function pointer */
    void (*func_ptr) (int) = bad_sink;
    /* Initialize data */
    data = -1;
    /* FLAW: Use a number larger than SHRT_MAX */
    data = SHRT_MAX + 5;
    /* use the function pointer */
    func_ptr(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() uses the GoodSource with the BadSink */
static void goodG2B_sink(int data)
{
    {
        /* POTENTIAL FLAW: Convert data to a char, possibly causing a truncation error */
        char c = (char)data;
        printHexCharLine(c);
    }
}

static void goodG2B()
{
    int data;
    void (*func_ptr) (int) = goodG2B_sink;
    /* Initialize data */
    data = -1;
    /* FIX: Use a positive integer less than CHAR_MAX*/
    data = CHAR_MAX-5;
    func_ptr(data);
}

void CWE197_Numeric_Truncation_Error__int_large_to_char_44_good()
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
    CWE197_Numeric_Truncation_Error__int_large_to_char_44_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE197_Numeric_Truncation_Error__int_large_to_char_44_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
