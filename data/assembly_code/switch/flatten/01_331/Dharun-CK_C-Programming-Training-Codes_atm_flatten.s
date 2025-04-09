	.file	"Dharun-CK_C-Programming-Training-Codes_atm_flatten.c"
	.text
	.globl	_TIG_IZ_DFVK_envp
	.bss
	.align 8
	.type	_TIG_IZ_DFVK_envp, @object
	.size	_TIG_IZ_DFVK_envp, 8
_TIG_IZ_DFVK_envp:
	.zero	8
	.globl	_TIG_IZ_DFVK_argv
	.align 8
	.type	_TIG_IZ_DFVK_argv, @object
	.size	_TIG_IZ_DFVK_argv, 8
_TIG_IZ_DFVK_argv:
	.zero	8
	.globl	_TIG_IZ_DFVK_argc
	.align 4
	.type	_TIG_IZ_DFVK_argc, @object
	.size	_TIG_IZ_DFVK_argc, 4
_TIG_IZ_DFVK_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"MISSMATCHED"
.LC1:
	.string	"Enter the Amount:"
.LC2:
	.string	"%li"
.LC3:
	.string	"THE NEW BALANCE is %li"
.LC4:
	.string	" The account Number IS %li\n"
.LC5:
	.string	"The Balance is %li\n"
	.align 8
.LC6:
	.string	"\n1 - WITHDRAW\n2- DEPOSIT\n3- BALANCE ENQUIRY\n4- PIN CHANGE"
.LC7:
	.string	"\nEnter THE Opinion:"
.LC8:
	.string	"AMOUNT EXCEEDED!"
	.align 8
.LC9:
	.string	"***Please Check the Account Number***"
.LC10:
	.string	"Enter The New Pin :"
.LC11:
	.string	"Re-enter The Pin:"
.LC12:
	.string	"Enter The pin: "
.LC13:
	.string	"Enter THe Old pin : "
.LC14:
	.string	"Your New pin is %li"
.LC15:
	.string	"Enter the PIN:"
	.align 8
.LC16:
	.string	"****************************************************************************"
.LC17:
	.string	"\nA*T*M     TRANSACTIONS "
	.align 8
.LC18:
	.string	"\n****************************************************************************"
.LC19:
	.string	"\nEnter the Account Number:"
.LC20:
	.string	"Invalid Pin!"
.LC21:
	.string	"Incorrect Pin!"
.LC22:
	.string	"\nVerifying..........."
.LC23:
	.string	"\nAccount Number :%li"
.LC24:
	.string	"\nBalance : %li"
	.align 8
.LC25:
	.string	"\nThe Balance is Verified! properly"
	.text
	.globl	main
	.type	main, @function
main:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	addq	$-128, %rsp
	movl	%edi, -100(%rbp)
	movq	%rsi, -112(%rbp)
	movq	%rdx, -120(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_DFVK_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_DFVK_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_DFVK_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 109 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-DFVK--0
# 0 "" 2
#NO_APP
	movl	-100(%rbp), %eax
	movl	%eax, _TIG_IZ_DFVK_argc(%rip)
	movq	-112(%rbp), %rax
	movq	%rax, _TIG_IZ_DFVK_argv(%rip)
	movq	-120(%rbp), %rax
	movq	%rax, _TIG_IZ_DFVK_envp(%rip)
	nop
	movq	$10, -24(%rbp)
.L53:
	cmpq	$34, -24(%rbp)
	ja	.L56
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L8(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L8(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L8:
	.long	.L56-.L8
	.long	.L56-.L8
	.long	.L32-.L8
	.long	.L56-.L8
	.long	.L56-.L8
	.long	.L31-.L8
	.long	.L30-.L8
	.long	.L29-.L8
	.long	.L56-.L8
	.long	.L28-.L8
	.long	.L27-.L8
	.long	.L26-.L8
	.long	.L25-.L8
	.long	.L24-.L8
	.long	.L23-.L8
	.long	.L56-.L8
	.long	.L56-.L8
	.long	.L22-.L8
	.long	.L21-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L56-.L8
	.long	.L56-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L56-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L21:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$7, -24(%rbp)
	jmp	.L33
.L11:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-56(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	-56(%rbp), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$13, -24(%rbp)
	jmp	.L33
.L23:
	movq	-96(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-40(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$10, %edi
	call	putchar@PLT
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-64(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$33, -24(%rbp)
	jmp	.L33
.L25:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$13, -24(%rbp)
	jmp	.L33
.L16:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$13, -24(%rbp)
	jmp	.L33
.L18:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-80(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-72(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$11, -24(%rbp)
	jmp	.L33
.L15:
	movq	-56(%rbp), %rax
	cmpq	%rax, -40(%rbp)
	jle	.L34
	movq	$5, -24(%rbp)
	jmp	.L33
.L34:
	movq	$12, -24(%rbp)
	jmp	.L33
.L26:
	movq	-80(%rbp), %rdx
	movq	-72(%rbp), %rax
	cmpq	%rax, %rdx
	jne	.L36
	movq	$32, -24(%rbp)
	jmp	.L33
.L36:
	movq	$18, -24(%rbp)
	jmp	.L33
.L28:
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-88(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$29, -24(%rbp)
	jmp	.L33
.L24:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L54
	jmp	.L55
.L20:
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-88(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$22, -24(%rbp)
	jmp	.L33
.L10:
	movq	-80(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$7, -24(%rbp)
	jmp	.L33
.L22:
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-88(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$28, -24(%rbp)
	jmp	.L33
.L30:
	movabsq	$732121205016, %rax
	movq	%rax, -48(%rbp)
	movq	$120000, -40(%rbp)
	movq	$7321, -32(%rbp)
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$10, %edi
	call	putchar@PLT
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC18(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC19(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-96(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$27, -24(%rbp)
	jmp	.L33
.L14:
	movq	-96(%rbp), %rax
	cmpq	%rax, -48(%rbp)
	je	.L39
	movq	$23, -24(%rbp)
	jmp	.L33
.L39:
	movq	$17, -24(%rbp)
	jmp	.L33
.L7:
	leaq	.LC20(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$13, -24(%rbp)
	jmp	.L33
.L17:
	movq	-88(%rbp), %rax
	cmpq	%rax, -32(%rbp)
	jne	.L41
	movq	$21, -24(%rbp)
	jmp	.L33
.L41:
	movq	$2, -24(%rbp)
	jmp	.L33
.L13:
	movq	-88(%rbp), %rax
	cmpq	%rax, -32(%rbp)
	je	.L43
	movq	$34, -24(%rbp)
	jmp	.L33
.L43:
	movq	$14, -24(%rbp)
	jmp	.L33
.L31:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-56(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	-56(%rbp), %rdx
	movq	-40(%rbp), %rax
	subq	%rdx, %rax
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$13, -24(%rbp)
	jmp	.L33
.L9:
	movq	-64(%rbp), %rax
	cmpq	$4, %rax
	je	.L45
	cmpq	$4, %rax
	jg	.L46
	cmpq	$3, %rax
	je	.L47
	cmpq	$3, %rax
	jg	.L46
	cmpq	$1, %rax
	je	.L48
	cmpq	$2, %rax
	je	.L49
	jmp	.L46
.L45:
	movq	$9, -24(%rbp)
	jmp	.L50
.L47:
	movq	$20, -24(%rbp)
	jmp	.L50
.L49:
	movq	$30, -24(%rbp)
	jmp	.L50
.L48:
	movq	$26, -24(%rbp)
	jmp	.L50
.L46:
	movq	$7, -24(%rbp)
	nop
.L50:
	jmp	.L33
.L27:
	movq	$6, -24(%rbp)
	jmp	.L33
.L29:
	movq	$13, -24(%rbp)
	jmp	.L33
.L12:
	movq	-88(%rbp), %rax
	cmpq	%rax, -32(%rbp)
	jne	.L51
	movq	$19, -24(%rbp)
	jmp	.L33
.L51:
	movq	$7, -24(%rbp)
	jmp	.L33
.L32:
	leaq	.LC21(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$7, -24(%rbp)
	jmp	.L33
.L19:
	leaq	.LC22(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-48(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC23(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-40(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC24(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC25(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$13, -24(%rbp)
	jmp	.L33
.L56:
	nop
.L33:
	jmp	.L53
.L55:
	call	__stack_chk_fail@PLT
.L54:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	main, .-main
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
