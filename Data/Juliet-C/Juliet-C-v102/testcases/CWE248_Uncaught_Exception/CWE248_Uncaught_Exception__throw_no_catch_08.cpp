/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE248_Uncaught_Exception__throw_no_catch_08.cpp
Label Definition File: CWE248_Uncaught_Exception.label.xml
Template File: point-flaw-08.tmpl.cpp
*/
/*
 * @description
 * CWE: 248 Uncaught Exception
 * Sinks: throw_no_catch
 *    GoodSink: Throw and catch an exception
 *    BadSink : Throw an exception
 * Flow Variant: 08 Control flow: if(static_returns_t()) and if(static_returns_f())
 *
 * */

#include "std_testcase.h"

#include <stdexcept>

using namespace std;

/* The two function below always return the same value, so a tool
   should be able to identify that calls to the functions will always
   return a fixed value. */
static int static_returns_t()
{
    return 1;
}

static int static_returns_f()
{
    return 0;
}

namespace CWE248_Uncaught_Exception__throw_no_catch_08
{

#ifndef OMITBAD

void bad()
{
    if(static_returns_t())
    {
        {
            /* FLAW: Throw an exception, but do not catch */
            throw 25;
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            try
            {
                throw 25;
            }
            /* FIX: Catch the exception throw in the try block */
            catch (int e)
            {
                printLine("Exception caught - value: ");
                printIntLine(e);
            }
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good1() uses if(static_returns_f()) instead of if(static_returns_t()) */
static void good1()
{
    if(static_returns_f())
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* FLAW: Throw an exception, but do not catch */
            throw 25;
        }
    }
    else
    {
        {
            try
            {
                throw 25;
            }
            /* FIX: Catch the exception throw in the try block */
            catch (int e)
            {
                printLine("Exception caught - value: ");
                printIntLine(e);
            }
        }
    }
}

/* good2() reverses the bodies in the if statement */
static void good2()
{
    if(static_returns_t())
    {
        {
            try
            {
                throw 25;
            }
            /* FIX: Catch the exception throw in the try block */
            catch (int e)
            {
                printLine("Exception caught - value: ");
                printIntLine(e);
            }
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* FLAW: Throw an exception, but do not catch */
            throw 25;
        }
    }
}

void good()
{
    good1();
    good2();
}

#endif /* OMITGOOD */

} // close namespace

/* Below is the main(). It is only used when building this testcase on
   its own for testing or for building a binary to use in testing binary
   analysis tools. It is not used when compiling all the testcases as one
   application, which is how source code analysis tools are tested. */

#ifdef INCLUDEMAIN

using namespace CWE248_Uncaught_Exception__throw_no_catch_08; // so that we can use good and bad easily

int main(int argc, char * argv[])
{
    /* seed randomness */
    srand( (unsigned)time(NULL) );
#ifndef OMITGOOD
    printLine("Calling good()...");
    good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
