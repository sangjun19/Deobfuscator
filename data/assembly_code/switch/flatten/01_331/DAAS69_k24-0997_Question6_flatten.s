	.file	"DAAS69_k24-0997_Question6_flatten.c"
	.text
	.globl	_TIG_IZ_FG2G_argv
	.bss
	.align 8
	.type	_TIG_IZ_FG2G_argv, @object
	.size	_TIG_IZ_FG2G_argv, 8
_TIG_IZ_FG2G_argv:
	.zero	8
	.globl	_TIG_IZ_FG2G_argc
	.align 4
	.type	_TIG_IZ_FG2G_argc, @object
	.size	_TIG_IZ_FG2G_argc, 4
_TIG_IZ_FG2G_argc:
	.zero	4
	.globl	_TIG_IZ_FG2G_envp
	.align 8
	.type	_TIG_IZ_FG2G_envp, @object
	.size	_TIG_IZ_FG2G_envp, 8
_TIG_IZ_FG2G_envp:
	.zero	8
	.section	.rodata
	.align 8
.LC0:
	.string	"Sorry, only %d seat(s) are available for this package.\n\n"
	.align 8
.LC1:
	.string	"Invalid package choice! Please try again."
.LC2:
	.string	"Available packages:"
	.align 8
.LC3:
	.string	"Enter the package number you want to book: "
.LC4:
	.string	"%d"
	.align 8
.LC5:
	.string	"Enter the number of seats you want to book: "
	.align 8
.LC6:
	.string	"Successfully booked %d seat(s) for package '%s'!\n\n"
	.text
	.globl	book_package
	.type	book_package, @function
book_package:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movl	%esi, -44(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$3, -16(%rbp)
.L23:
	cmpq	$14, -16(%rbp)
	ja	.L26
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L4(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L4(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L4:
	.long	.L14-.L4
	.long	.L13-.L4
	.long	.L12-.L4
	.long	.L11-.L4
	.long	.L26-.L4
	.long	.L26-.L4
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L27-.L4
	.long	.L26-.L4
	.long	.L6-.L4
	.long	.L26-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L3:
	movl	-24(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	leaq	-40(%rax), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	32(%rax), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$9, -16(%rbp)
	jmp	.L15
.L8:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-40(%rbp), %rax
	movl	$3, %esi
	movq	%rax, %rdi
	call	book_package
	movq	$13, -16(%rbp)
	jmp	.L15
.L13:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	display
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$7, -16(%rbp)
	jmp	.L15
.L11:
	movq	$1, -16(%rbp)
	jmp	.L15
.L6:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-40(%rbp), %rax
	movl	$3, %esi
	movq	%rax, %rdi
	call	book_package
	movq	$13, -16(%rbp)
	jmp	.L15
.L5:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$6, -16(%rbp)
	jmp	.L15
.L10:
	movl	-24(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	leaq	-40(%rax), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	32(%rax), %edx
	movl	-20(%rbp), %eax
	cmpl	%eax, %edx
	jl	.L17
	movq	$2, -16(%rbp)
	jmp	.L15
.L17:
	movq	$14, -16(%rbp)
	jmp	.L15
.L14:
	movl	-24(%rbp), %eax
	cmpl	%eax, -44(%rbp)
	jge	.L19
	movq	$11, -16(%rbp)
	jmp	.L15
.L19:
	movq	$13, -16(%rbp)
	jmp	.L15
.L9:
	movl	-24(%rbp), %eax
	testl	%eax, %eax
	jg	.L21
	movq	$8, -16(%rbp)
	jmp	.L15
.L21:
	movq	$0, -16(%rbp)
	jmp	.L15
.L12:
	movl	-24(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	leaq	-40(%rax), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	32(%rax), %ecx
	movl	-20(%rbp), %esi
	movl	-24(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	leaq	-40(%rax), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	subl	%esi, %ecx
	movl	%ecx, %edx
	movl	%edx, 32(%rax)
	movl	-24(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	leaq	-40(%rax), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, %rdx
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$9, -16(%rbp)
	jmp	.L15
.L26:
	nop
.L15:
	jmp	.L23
.L27:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L25
	call	__stack_chk_fail@PLT
.L25:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	book_package, .-book_package
	.section	.rodata
.LC7:
	.string	"exiting :)"
	.align 8
.LC8:
	.string	"Invalid choice! Please try again."
.LC9:
	.string	"Welcome to Fast Travels!"
.LC10:
	.string	"1. Display available packages"
.LC11:
	.string	"2. Book a package"
.LC12:
	.string	"3. Exit"
.LC13:
	.string	"Enter your choice: "
	.text
	.globl	main
	.type	main, @function
main:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$192, %rsp
	movl	%edi, -164(%rbp)
	movq	%rsi, -176(%rbp)
	movq	%rdx, -184(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_FG2G_envp(%rip)
	nop
.L29:
	movq	$0, _TIG_IZ_FG2G_argv(%rip)
	nop
.L30:
	movl	$0, _TIG_IZ_FG2G_argc(%rip)
	nop
	nop
.L31:
.L32:
#APP
# 89 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-FG2G--0
# 0 "" 2
#NO_APP
	movl	-164(%rbp), %eax
	movl	%eax, _TIG_IZ_FG2G_argc(%rip)
	movq	-176(%rbp), %rax
	movq	%rax, _TIG_IZ_FG2G_argv(%rip)
	movq	-184(%rbp), %rax
	movq	%rax, _TIG_IZ_FG2G_envp(%rip)
	nop
	movq	$5, -136(%rbp)
.L66:
	cmpq	$32, -136(%rbp)
	ja	.L69
	movq	-136(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L35(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L35(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L35:
	.long	.L69-.L35
	.long	.L52-.L35
	.long	.L51-.L35
	.long	.L69-.L35
	.long	.L69-.L35
	.long	.L50-.L35
	.long	.L69-.L35
	.long	.L49-.L35
	.long	.L69-.L35
	.long	.L69-.L35
	.long	.L48-.L35
	.long	.L47-.L35
	.long	.L69-.L35
	.long	.L69-.L35
	.long	.L69-.L35
	.long	.L46-.L35
	.long	.L45-.L35
	.long	.L44-.L35
	.long	.L43-.L35
	.long	.L42-.L35
	.long	.L41-.L35
	.long	.L69-.L35
	.long	.L40-.L35
	.long	.L39-.L35
	.long	.L69-.L35
	.long	.L69-.L35
	.long	.L69-.L35
	.long	.L69-.L35
	.long	.L69-.L35
	.long	.L38-.L35
	.long	.L37-.L35
	.long	.L36-.L35
	.long	.L34-.L35
	.text
.L43:
	movl	-140(%rbp), %eax
	movb	$0, -39(%rbp,%rax)
	addl	$1, -140(%rbp)
	movq	$29, -136(%rbp)
	jmp	.L53
.L37:
	movb	$112, -128(%rbp)
	movb	$97, -127(%rbp)
	movb	$99, -126(%rbp)
	movb	$107, -125(%rbp)
	movb	$97, -124(%rbp)
	movb	$103, -123(%rbp)
	movb	$101, -122(%rbp)
	movb	$49, -121(%rbp)
	movb	$0, -120(%rbp)
	movb	$85, -119(%rbp)
	movb	$83, -118(%rbp)
	movb	$65, -117(%rbp)
	movb	$0, -116(%rbp)
	movl	$4, -148(%rbp)
	movq	$22, -136(%rbp)
	jmp	.L53
.L46:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L67
	jmp	.L68
.L36:
	movl	$10, -108(%rbp)
	movq	$600000, -104(%rbp)
	movl	$21, -96(%rbp)
	movb	$112, -88(%rbp)
	movb	$97, -87(%rbp)
	movb	$99, -86(%rbp)
	movb	$107, -85(%rbp)
	movb	$97, -84(%rbp)
	movb	$103, -83(%rbp)
	movb	$101, -82(%rbp)
	movb	$50, -81(%rbp)
	movb	$0, -80(%rbp)
	movb	$82, -79(%rbp)
	movb	$117, -78(%rbp)
	movb	$115, -77(%rbp)
	movb	$115, -76(%rbp)
	movb	$105, -75(%rbp)
	movb	$97, -74(%rbp)
	movb	$0, -73(%rbp)
	movl	$7, -144(%rbp)
	movq	$10, -136(%rbp)
	jmp	.L53
.L52:
	movl	-152(%rbp), %eax
	cmpl	$3, %eax
	je	.L55
	cmpl	$3, %eax
	jg	.L56
	cmpl	$1, %eax
	je	.L57
	cmpl	$2, %eax
	je	.L58
	jmp	.L56
.L55:
	movq	$11, -136(%rbp)
	jmp	.L59
.L58:
	movq	$17, -136(%rbp)
	jmp	.L59
.L57:
	movq	$2, -136(%rbp)
	jmp	.L59
.L56:
	movq	$19, -136(%rbp)
	nop
.L59:
	jmp	.L53
.L39:
	movl	-148(%rbp), %eax
	movb	$0, -119(%rbp,%rax)
	addl	$1, -148(%rbp)
	movq	$22, -136(%rbp)
	jmp	.L53
.L45:
	movl	$14, -28(%rbp)
	movq	$200000, -24(%rbp)
	movl	$5, -16(%rbp)
	movq	$20, -136(%rbp)
	jmp	.L53
.L47:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$15, -136(%rbp)
	jmp	.L53
.L42:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$20, -136(%rbp)
	jmp	.L53
.L34:
	movl	$12, -68(%rbp)
	movq	$400000, -64(%rbp)
	movl	$32, -56(%rbp)
	movb	$112, -48(%rbp)
	movb	$97, -47(%rbp)
	movb	$99, -46(%rbp)
	movb	$107, -45(%rbp)
	movb	$97, -44(%rbp)
	movb	$103, -43(%rbp)
	movb	$101, -42(%rbp)
	movb	$51, -41(%rbp)
	movb	$0, -40(%rbp)
	movb	$67, -39(%rbp)
	movb	$104, -38(%rbp)
	movb	$105, -37(%rbp)
	movb	$110, -36(%rbp)
	movb	$97, -35(%rbp)
	movb	$0, -34(%rbp)
	movl	$6, -140(%rbp)
	movq	$29, -136(%rbp)
	jmp	.L53
.L44:
	leaq	-128(%rbp), %rax
	movl	$3, %esi
	movq	%rax, %rdi
	call	book_package
	movq	$20, -136(%rbp)
	jmp	.L53
.L40:
	cmpl	$8, -148(%rbp)
	jbe	.L60
	movq	$31, -136(%rbp)
	jmp	.L53
.L60:
	movq	$23, -136(%rbp)
	jmp	.L53
.L50:
	movq	$30, -136(%rbp)
	jmp	.L53
.L48:
	cmpl	$8, -144(%rbp)
	jbe	.L62
	movq	$32, -136(%rbp)
	jmp	.L53
.L62:
	movq	$7, -136(%rbp)
	jmp	.L53
.L49:
	movl	-144(%rbp), %eax
	movb	$0, -79(%rbp,%rax)
	addl	$1, -144(%rbp)
	movq	$10, -136(%rbp)
	jmp	.L53
.L38:
	cmpl	$8, -140(%rbp)
	jbe	.L64
	movq	$16, -136(%rbp)
	jmp	.L53
.L64:
	movq	$18, -136(%rbp)
	jmp	.L53
.L51:
	leaq	-128(%rbp), %rax
	movq	%rax, %rdi
	call	display
	movq	$20, -136(%rbp)
	jmp	.L53
.L41:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-152(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$1, -136(%rbp)
	jmp	.L53
.L69:
	nop
.L53:
	jmp	.L66
.L68:
	call	__stack_chk_fail@PLT
.L67:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	main, .-main
	.section	.rodata
.LC14:
	.string	"Package number %d:\n"
.LC15:
	.string	"Name: %s\n"
.LC16:
	.string	"Destination: %s\n"
.LC17:
	.string	"Duration: %d days\n"
.LC18:
	.string	"Cost: %ld\n"
.LC19:
	.string	"Available seats: %d\n\n"
	.text
	.globl	display
	.type	display, @function
display:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$5, -8(%rbp)
.L79:
	cmpq	$6, -8(%rbp)
	je	.L80
	cmpq	$6, -8(%rbp)
	ja	.L81
	cmpq	$5, -8(%rbp)
	je	.L73
	cmpq	$5, -8(%rbp)
	ja	.L81
	cmpq	$2, -8(%rbp)
	je	.L74
	cmpq	$3, -8(%rbp)
	jne	.L81
	movl	-12(%rbp), %eax
	addl	$1, %eax
	movl	%eax, %esi
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-12(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, %rsi
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-12(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	addq	$9, %rax
	movq	%rax, %rsi
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-12(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	20(%rax), %eax
	movl	%eax, %esi
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-12(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	24(%rax), %rax
	movq	%rax, %rsi
	leaq	.LC18(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-12(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	32(%rax), %eax
	movl	%eax, %esi
	leaq	.LC19(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L75
.L73:
	movl	$0, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L75
.L74:
	cmpl	$2, -12(%rbp)
	jg	.L77
	movq	$3, -8(%rbp)
	jmp	.L75
.L77:
	movq	$6, -8(%rbp)
	jmp	.L75
.L81:
	nop
.L75:
	jmp	.L79
.L80:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	display, .-display
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
