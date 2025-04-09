	.file	"ebrurazi_BasitATM_main_flatten.c"
	.text
	.globl	_TIG_IZ_evCy_argc
	.bss
	.align 4
	.type	_TIG_IZ_evCy_argc, @object
	.size	_TIG_IZ_evCy_argc, 4
_TIG_IZ_evCy_argc:
	.zero	4
	.globl	_TIG_IZ_evCy_envp
	.align 8
	.type	_TIG_IZ_evCy_envp, @object
	.size	_TIG_IZ_evCy_envp, 8
_TIG_IZ_evCy_envp:
	.zero	8
	.globl	_TIG_IZ_evCy_argv
	.align 8
	.type	_TIG_IZ_evCy_argv, @object
	.size	_TIG_IZ_evCy_argv, 8
_TIG_IZ_evCy_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"\304\260\305\237leminizi giriniz: "
.LC1:
	.string	"%d"
.LC2:
	.string	"\304\260yi g\303\274nler..."
.LC3:
	.string	"Bakiyeniz: %d\n"
	.align 8
.LC4:
	.string	"Yat\304\261rmak istedi\304\237iniz tutari giriniz : "
.LC5:
	.string	"Yeni bakiyeniz : %d\n"
.LC6:
	.string	"Bakiyeniz: %d"
	.align 8
.LC7:
	.string	"\n\n*****************\304\260\305\236LEMLER*****************"
	.align 8
.LC8:
	.string	"1. Para \303\207ekme\n2. Para Yat\304\261rma\n3. Para Bakiye Sorgulama\n4. Kart \304\260ade\n"
	.align 8
.LC9:
	.string	"\303\207ekmek istedi\304\237iniz tutari giriniz : "
.LC10:
	.string	"Yanl\304\261\305\237 girdiniz!!"
.LC11:
	.string	"Bakiyeniz: %d\n\n"
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
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_evCy_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_evCy_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_evCy_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 109 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-evCy--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_evCy_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_evCy_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_evCy_envp(%rip)
	nop
	movq	$5, -16(%rbp)
.L32:
	cmpq	$20, -16(%rbp)
	ja	.L35
	movq	-16(%rbp), %rax
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
	.long	.L20-.L8
	.long	.L35-.L8
	.long	.L19-.L8
	.long	.L35-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L35-.L8
	.long	.L35-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L35-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L35-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L35-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L35-.L8
	.long	.L7-.L8
	.text
.L9:
	movl	-28(%rbp), %eax
	cmpl	$4, %eax
	je	.L21
	movq	$0, -16(%rbp)
	jmp	.L23
.L21:
	movq	$14, -16(%rbp)
	jmp	.L23
.L18:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$18, -16(%rbp)
	jmp	.L23
.L12:
	movl	-28(%rbp), %eax
	cmpl	$4, %eax
	jne	.L24
	movq	$12, -16(%rbp)
	jmp	.L23
.L24:
	movq	$15, -16(%rbp)
	jmp	.L23
.L11:
	movl	$10, %edi
	call	putchar@PLT
	movq	$2, -16(%rbp)
	jmp	.L23
.L13:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$15, -16(%rbp)
	jmp	.L23
.L16:
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-24(%rbp), %eax
	addl	%eax, -20(%rbp)
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$4, -16(%rbp)
	jmp	.L23
.L14:
	movl	$1000, -20(%rbp)
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$18, -16(%rbp)
	jmp	.L23
.L15:
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-24(%rbp), %eax
	subl	%eax, -20(%rbp)
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$4, -16(%rbp)
	jmp	.L23
.L10:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$4, -16(%rbp)
	jmp	.L23
.L17:
	movq	$11, -16(%rbp)
	jmp	.L23
.L20:
	movl	-28(%rbp), %eax
	cmpl	$3, %eax
	je	.L26
	cmpl	$3, %eax
	jg	.L27
	cmpl	$1, %eax
	je	.L28
	cmpl	$2, %eax
	je	.L29
	jmp	.L27
.L26:
	movq	$20, -16(%rbp)
	jmp	.L30
.L29:
	movq	$8, -16(%rbp)
	jmp	.L30
.L28:
	movq	$9, -16(%rbp)
	jmp	.L30
.L27:
	movq	$17, -16(%rbp)
	nop
.L30:
	jmp	.L23
.L19:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L33
	jmp	.L34
.L7:
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$4, -16(%rbp)
	jmp	.L23
.L35:
	nop
.L23:
	jmp	.L32
.L34:
	call	__stack_chk_fail@PLT
.L33:
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
