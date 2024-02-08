/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE190_Integer_Overflow__unsigned_char_fixed_square_44.c
Label Definition File: CWE190_Integer_Overflow.label.xml
Template File: sources-sinks-44.tmpl.c
*/
/*
 * @description
 * CWE: 190 Integer Overflow
 * BadSource: fixed Fixed value
 * GoodSource: Small, non-zero
 * Sinks: square
 *    GoodSink: Ensure there is no overflow before performing the squaring operation
 *    BadSink : Square data
 * Flow Variant: 44 Data/control flow: data passed as an argument from one function to a function in the same source file called via a function pointer
 *
 * */

#include "std_testcase.h"

#include <math.h>

#ifndef OMITBAD

static void bad_sink(unsigned char data)
{
    {
        /* POTENTIAL FLAW: Squaring data could cause an overflow */
        unsigned char result = data * data;
        printHexUnsignedCharLine(result);
    }
}

void CWE190_Integer_Overflow__unsigned_char_fixed_square_44_bad()
{
    unsigned char data;
    /* define a function pointer */
    void (*func_ptr) (unsigned char) = bad_sink;
    data = ' ';
    /* FLAW: Use the maximum size of the data type */
    data = UCHAR_MAX;
    /* use the function pointer */
    func_ptr(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() uses the GoodSource with the BadSink */
static void goodG2B_sink(unsigned char data)
{
    {
        /* POTENTIAL FLAW: Squaring data could cause an overflow */
        unsigned char result = data * data;
        printHexUnsignedCharLine(result);
    }
}

static void goodG2B()
{
    unsigned char data;
    void (*func_ptr) (unsigned char) = goodG2B_sink;
    data = ' ';
    /* FIX: Use a small, non-zero value that will not cause an overflow in the sinks */
    data = 5;
    func_ptr(data);
}

/* goodB2G() uses the BadSource with the GoodSink */
static void goodB2G_sink(unsigned char data)
{
    {
        unsigned char result = -1;
        /* FIX: Add a check to prevent an overflow from occurring */
        if (data <= (unsigned char)sqrt((unsigned char)UCHAR_MAX))
        {
            result = data * data;
            printHexUnsignedCharLine(result);
        }
        else
        {
            printLine("Input value is too large to perform arithmetic safely.");
        }
    }
}

static void goodB2G()
{
    unsigned char data;
    void (*func_ptr) (unsigned char) = goodB2G_sink;
    data = ' ';
    /* FLAW: Use the maximum size of the data type */
    data = UCHAR_MAX;
    func_ptr(data);
}

void CWE190_Integer_Overflow__unsigned_char_fixed_square_44_good()
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
    CWE190_Integer_Overflow__unsigned_char_fixed_square_44_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE190_Integer_Overflow__unsigned_char_fixed_square_44_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
