/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE396_Catch_Generic_Exception__dotdotdot_17.cpp
Label Definition File: CWE396_Catch_Generic_Exception.label.xml
Template File: point-flaw-17.tmpl.cpp
*/
/*
 * @description
 * CWE: 396 Catch Generic Exception
 * Sinks: dotdotdot
 *    GoodSink: Catch specific exceptions
 *    BadSink : Catch ..., which will catch any exception
 * Flow Variant: 17 Control flow: for loops
 *
 * */

#include "std_testcase.h"

#include <stdexcept> /* for out_of_range and domain_error */

using namespace std; /* allow easy references to out_of_range and domain_error */

namespace CWE396_Catch_Generic_Exception__dotdotdot_17
{

#ifndef OMITBAD

void bad()
{
    int j,k;
    for(j = 0; j < 1; j++)
    {
        try
        {
            if (rand()%2 == 0) throw out_of_range("err1");
            if (rand()%2 == 0) throw domain_error("err2");
        }
        catch (...)
        {
            /* FLAW: this catches _everything_, STL exceptions or otherwise */
            printLine("exception");
        }
        printLine("ok");
    }
    for(k = 0; k < 0; k++)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        try
        {
            if (rand()%2 == 0) throw out_of_range("err1");
            if (rand()%2 == 0) throw domain_error("err2");
        }
        catch (out_of_range &)
        {
            /* FIX: specify each catch individually */
            printLine("out_of_range");
        }
        catch (domain_error &)
        {
            printLine("domain_error");
            return;
        }
        printLine("ok");
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good1() changes the conditions on the for statements */
static void good1()
{
    int j,k;
    for(j = 0; j < 0; j++)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        try
        {
            if (rand()%2 == 0) throw out_of_range("err1");
            if (rand()%2 == 0) throw domain_error("err2");
        }
        catch (...)
        {
            /* FLAW: this catches _everything_, STL exceptions or otherwise */
            printLine("exception");
        }
        printLine("ok");
    }
    for(k = 0; k < 1; k++)
    {
        try
        {
            if (rand()%2 == 0) throw out_of_range("err1");
            if (rand()%2 == 0) throw domain_error("err2");
        }
        catch (out_of_range &)
        {
            /* FIX: specify each catch individually */
            printLine("out_of_range");
        }
        catch (domain_error &)
        {
            printLine("domain_error");
            return;
        }
        printLine("ok");
    }
}

void good()
{
    good1();
}

#endif /* OMITGOOD */

} // close namespace

/* Below is the main(). It is only used when building this testcase on
   its own for testing or for building a binary to use in testing binary
   analysis tools. It is not used when compiling all the testcases as one
   application, which is how source code analysis tools are tested. */

#ifdef INCLUDEMAIN

using namespace CWE396_Catch_Generic_Exception__dotdotdot_17; // so that we can use good and bad easily

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
