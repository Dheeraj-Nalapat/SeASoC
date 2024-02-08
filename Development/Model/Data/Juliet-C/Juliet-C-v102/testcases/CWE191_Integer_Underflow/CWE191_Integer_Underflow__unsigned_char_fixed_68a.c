/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE191_Integer_Underflow__unsigned_char_fixed_68a.c
Label Definition File: CWE191_Integer_Underflow.label.xml
Template File: sources-sinks-68a.tmpl.c
*/
/*
 * @description
 * CWE: 191 Integer Underflow
 * BadSource: fixed Fixed value
 * GoodSource: Small, non-zero
 * Sinks:
 *    GoodSink: Ensure there is no underflow before performing the subtraction
 *    BadSink : Subtract 1 from data
 * Flow Variant: 68 Data flow: data passed as a global variable from one function to another in different source files
 *
 * */

#include "std_testcase.h"

unsigned char CWE191_Integer_Underflow__unsigned_char_fixed_68_bad_data;
unsigned char CWE191_Integer_Underflow__unsigned_char_fixed_68_goodG2B_data;
unsigned char CWE191_Integer_Underflow__unsigned_char_fixed_68_goodB2G_data;

#ifndef OMITBAD

/* bad function declaration */
void CWE191_Integer_Underflow__unsigned_char_fixed_68b_bad_sink();

void CWE191_Integer_Underflow__unsigned_char_fixed_68_bad()
{
    unsigned char data;
    data = ' ';
    /* FLAW: Use the minimum size of the data type */
    data = 0;
    CWE191_Integer_Underflow__unsigned_char_fixed_68_bad_data = data;
    CWE191_Integer_Underflow__unsigned_char_fixed_68b_bad_sink();
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good function declarations */
void CWE191_Integer_Underflow__unsigned_char_fixed_68b_goodG2B_sink();
void CWE191_Integer_Underflow__unsigned_char_fixed_68b_goodB2G_sink();

/* goodG2B uses the GoodSource with the BadSink */
static void goodG2B()
{
    unsigned char data;
    data = ' ';
    /* FIX: Use a small value greater than the min value for this data type */
    data = 5;
    CWE191_Integer_Underflow__unsigned_char_fixed_68_goodG2B_data = data;
    CWE191_Integer_Underflow__unsigned_char_fixed_68b_goodG2B_sink();
}

/* goodB2G uses the BadSource with the GoodSink */
static void goodB2G()
{
    unsigned char data;
    data = ' ';
    /* FLAW: Use the minimum size of the data type */
    data = 0;
    CWE191_Integer_Underflow__unsigned_char_fixed_68_goodB2G_data = data;
    CWE191_Integer_Underflow__unsigned_char_fixed_68b_goodB2G_sink();
}

void CWE191_Integer_Underflow__unsigned_char_fixed_68_good()
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
    CWE191_Integer_Underflow__unsigned_char_fixed_68_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE191_Integer_Underflow__unsigned_char_fixed_68_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
