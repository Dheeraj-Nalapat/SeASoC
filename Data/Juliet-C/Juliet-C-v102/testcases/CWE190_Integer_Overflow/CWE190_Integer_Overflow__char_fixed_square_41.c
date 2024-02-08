/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE190_Integer_Overflow__char_fixed_square_41.c
Label Definition File: CWE190_Integer_Overflow.label.xml
Template File: sources-sinks-41.tmpl.c
*/
/*
 * @description
 * CWE: 190 Integer Overflow
 * BadSource: fixed Fixed value
 * GoodSource: Small, non-zero
 * Sinks: square
 *    GoodSink: Ensure there is no overflow before performing the squaring operation
 *    BadSink : Square data
 * Flow Variant: 41 Data flow: data passed as an argument from one function to another in the same source file
 *
 * */

#include "std_testcase.h"

#include <math.h>

#ifndef OMITBAD

static void bad_sink(char data)
{
    {
        /* POTENTIAL FLAW: Squaring data could cause an overflow */
        char result = data * data;
        printHexCharLine(result);
    }
}

void CWE190_Integer_Overflow__char_fixed_square_41_bad()
{
    char data;
    data = ' ';
    /* FLAW: Use the maximum size of the data type */
    data = CHAR_MAX;
    bad_sink(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
static void goodG2B_sink(char data)
{
    {
        /* POTENTIAL FLAW: Squaring data could cause an overflow */
        char result = data * data;
        printHexCharLine(result);
    }
}

static void goodG2B()
{
    char data;
    data = ' ';
    /* FIX: Use a small, non-zero value that will not cause an overflow in the sinks */
    data = 5;
    goodG2B_sink(data);
}

/* goodB2G uses the BadSource with the GoodSink */
static void goodB2G_sink(char data)
{
    {
        char result = -1;
        /* FIX: Add a check to prevent an overflow from occurring */
        if (data <= (char)sqrt((char)CHAR_MAX))
        {
            result = data * data;
            printHexCharLine(result);
        }
        else
        {
            printLine("Input value is too large to perform arithmetic safely.");
        }
    }
}

static void goodB2G()
{
    char data;
    data = ' ';
    /* FLAW: Use the maximum size of the data type */
    data = CHAR_MAX;
    goodB2G_sink(data);
}

void CWE190_Integer_Overflow__char_fixed_square_41_good()
{
    goodB2G();
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
    CWE190_Integer_Overflow__char_fixed_square_41_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE190_Integer_Overflow__char_fixed_square_41_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
