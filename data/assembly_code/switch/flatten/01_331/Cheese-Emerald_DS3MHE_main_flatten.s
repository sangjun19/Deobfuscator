	.file	"Cheese-Emerald_DS3MHE_main_flatten.c"
	.text
	.globl	_TIG_IZ_ro2U_envp
	.bss
	.align 8
	.type	_TIG_IZ_ro2U_envp, @object
	.size	_TIG_IZ_ro2U_envp, 8
_TIG_IZ_ro2U_envp:
	.zero	8
	.globl	_TIG_IZ_ro2U_argc
	.align 4
	.type	_TIG_IZ_ro2U_argc, @object
	.size	_TIG_IZ_ro2U_argc, 4
_TIG_IZ_ro2U_argc:
	.zero	4
	.globl	_TIG_IZ_ro2U_argv
	.align 8
	.type	_TIG_IZ_ro2U_argv, @object
	.size	_TIG_IZ_ro2U_argv, 8
_TIG_IZ_ro2U_argv:
	.zero	8
	.text
	.globl	check_s3m_header
	.type	check_s3m_header, @function
check_s3m_header:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	$9, -8(%rbp)
.L27:
	cmpq	$10, -8(%rbp)
	ja	.L28
	movq	-8(%rbp), %rax
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
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L10:
	movq	-24(%rbp), %rax
	addq	$47, %rax
	movzbl	(%rax), %eax
	cmpb	$77, %al
	je	.L15
	movq	$10, -8(%rbp)
	jmp	.L17
.L15:
	movq	$3, -8(%rbp)
	jmp	.L17
.L6:
	movl	$1, %eax
	jmp	.L18
.L13:
	movl	$1, %eax
	jmp	.L18
.L11:
	movl	$0, %eax
	jmp	.L18
.L5:
	cmpq	$0, -24(%rbp)
	jne	.L19
	movq	$6, -8(%rbp)
	jmp	.L17
.L19:
	movq	$5, -8(%rbp)
	jmp	.L17
.L8:
	movl	$2, %eax
	jmp	.L18
.L9:
	movq	-24(%rbp), %rax
	addq	$44, %rax
	movzbl	(%rax), %eax
	cmpb	$83, %al
	je	.L21
	movq	$1, -8(%rbp)
	jmp	.L17
.L21:
	movq	$0, -8(%rbp)
	jmp	.L17
.L3:
	movl	$1, %eax
	jmp	.L18
.L14:
	movq	-24(%rbp), %rax
	addq	$45, %rax
	movzbl	(%rax), %eax
	cmpb	$67, %al
	je	.L23
	movq	$2, -8(%rbp)
	jmp	.L17
.L23:
	movq	$7, -8(%rbp)
	jmp	.L17
.L7:
	movq	-24(%rbp), %rax
	addq	$46, %rax
	movzbl	(%rax), %eax
	cmpb	$82, %al
	je	.L25
	movq	$8, -8(%rbp)
	jmp	.L17
.L25:
	movq	$4, -8(%rbp)
	jmp	.L17
.L12:
	movl	$1, %eax
	jmp	.L18
.L28:
	nop
.L17:
	jmp	.L27
.L18:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	check_s3m_header, .-check_s3m_header
	.section	.rodata
.LC0:
	.string	"%1u"
	.align 8
.LC1:
	.string	"Would you like the song to be in stereo (1) or mono (0)?"
	.text
	.globl	handle_stereo_toggle
	.type	handle_stereo_toggle, @function
handle_stereo_toggle:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$8, -16(%rbp)
.L52:
	cmpq	$11, -16(%rbp)
	ja	.L55
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L32(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L32(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L32:
	.long	.L43-.L32
	.long	.L42-.L32
	.long	.L56-.L32
	.long	.L40-.L32
	.long	.L39-.L32
	.long	.L56-.L32
	.long	.L37-.L32
	.long	.L36-.L32
	.long	.L35-.L32
	.long	.L56-.L32
	.long	.L33-.L32
	.long	.L31-.L32
	.text
.L39:
	movl	$0, -20(%rbp)
	movq	$1, -16(%rbp)
	jmp	.L44
.L35:
	movl	$1, -32(%rbp)
	movq	$0, -16(%rbp)
	jmp	.L44
.L42:
	leaq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	%eax, -24(%rbp)
	movq	$11, -16(%rbp)
	jmp	.L44
.L40:
	movl	$1, -20(%rbp)
	movq	$1, -16(%rbp)
	jmp	.L44
.L31:
	cmpl	$1, -24(%rbp)
	jne	.L45
	movq	$10, -16(%rbp)
	jmp	.L44
.L45:
	movq	$5, -16(%rbp)
	jmp	.L44
.L37:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	%eax, -28(%rbp)
	movq	$7, -16(%rbp)
	jmp	.L44
.L33:
	movq	-40(%rbp), %rax
	addq	$51, %rax
	movzbl	(%rax), %eax
	movl	%eax, %edx
	movl	-32(%rbp), %eax
	movzbl	%al, %eax
	sall	$7, %eax
	orl	%eax, %edx
	movq	-40(%rbp), %rax
	addq	$51, %rax
	movb	%dl, (%rax)
	movq	$9, -16(%rbp)
	jmp	.L44
.L43:
	cmpq	$0, -40(%rbp)
	jne	.L48
	movq	$2, -16(%rbp)
	jmp	.L44
.L48:
	movq	$6, -16(%rbp)
	jmp	.L44
.L36:
	cmpl	$0, -28(%rbp)
	je	.L50
	movq	$4, -16(%rbp)
	jmp	.L44
.L50:
	movq	$3, -16(%rbp)
	jmp	.L44
.L55:
	nop
.L44:
	jmp	.L52
.L56:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L54
	call	__stack_chk_fail@PLT
.L54:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	handle_stereo_toggle, .-handle_stereo_toggle
	.section	.rodata
.LC2:
	.string	"%2X"
	.align 8
.LC3:
	.ascii	"\nThe bit meanings for the song flags (hex):\n0 (+1): ST2 vi"
	.ascii	"brato (deprecated)\n1 (+2): ST2 tempo (deprec"
	.string	"ated)\n2 (+4): Amiga slides (deprecated)\n3 (+8): 0-vol optimizations\n4 (+10): Enforce Amiga limits\n5 (+20): Enable SoundBlaster filter/FX (deprecated)\n6 (+40): Fast volume slides\n7 (+80): Pointer to special data is valid\n\nEnter your new value (hexadecimal):"
	.text
	.globl	handle_s3m_flags
	.type	handle_s3m_flags, @function
handle_s3m_flags:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$4, -16(%rbp)
.L80:
	cmpq	$11, -16(%rbp)
	ja	.L83
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L60(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L60(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L60:
	.long	.L84-.L60
	.long	.L70-.L60
	.long	.L69-.L60
	.long	.L68-.L60
	.long	.L67-.L60
	.long	.L84-.L60
	.long	.L65-.L60
	.long	.L64-.L60
	.long	.L63-.L60
	.long	.L62-.L60
	.long	.L84-.L60
	.long	.L59-.L60
	.text
.L67:
	movl	$0, -32(%rbp)
	movq	$11, -16(%rbp)
	jmp	.L72
.L63:
	leaq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	%eax, -24(%rbp)
	movq	$9, -16(%rbp)
	jmp	.L72
.L70:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	%eax, -28(%rbp)
	movq	$2, -16(%rbp)
	jmp	.L72
.L68:
	movl	-32(%rbp), %edx
	movq	-40(%rbp), %rax
	addq	$38, %rax
	movb	%dl, (%rax)
	movq	$5, -16(%rbp)
	jmp	.L72
.L59:
	cmpq	$0, -40(%rbp)
	jne	.L73
	movq	$0, -16(%rbp)
	jmp	.L72
.L73:
	movq	$1, -16(%rbp)
	jmp	.L72
.L62:
	cmpl	$1, -24(%rbp)
	jne	.L75
	movq	$3, -16(%rbp)
	jmp	.L72
.L75:
	movq	$10, -16(%rbp)
	jmp	.L72
.L65:
	movl	$0, -20(%rbp)
	movq	$8, -16(%rbp)
	jmp	.L72
.L64:
	movl	$1, -20(%rbp)
	movq	$8, -16(%rbp)
	jmp	.L72
.L69:
	cmpl	$0, -28(%rbp)
	je	.L78
	movq	$6, -16(%rbp)
	jmp	.L72
.L78:
	movq	$7, -16(%rbp)
	jmp	.L72
.L83:
	nop
.L72:
	jmp	.L80
.L84:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L82
	call	__stack_chk_fail@PLT
.L82:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	handle_s3m_flags, .-handle_s3m_flags
	.section	.rodata
.LC4:
	.string	"Song title: %.28s\n"
.LC5:
	.string	"Done!"
	.align 8
.LC6:
	.string	"Dumb S3M Header Editor\nby RepellantMold (2023, 2024)\n\n"
.LC7:
	.string	"Failed to open the file"
	.align 8
.LC8:
	.string	"Expected usage: %s <filename.s3m>"
.LC9:
	.string	"rb+"
.LC10:
	.string	"Too many arguments."
.LC11:
	.string	"Not a valid S3M file."
	.text
	.globl	main
	.type	main, @function
main:
.LFB8:
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
	movq	$0, _TIG_IZ_ro2U_envp(%rip)
	nop
.L86:
	movq	$0, _TIG_IZ_ro2U_argv(%rip)
	nop
.L87:
	movl	$0, _TIG_IZ_ro2U_argc(%rip)
	nop
	nop
.L88:
.L89:
#APP
# 232 "Cheese-Emerald_DS3MHE_main.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-ro2U--0
# 0 "" 2
#NO_APP
	movl	-164(%rbp), %eax
	movl	%eax, _TIG_IZ_ro2U_argc(%rip)
	movq	-176(%rbp), %rax
	movq	%rax, _TIG_IZ_ro2U_argv(%rip)
	movq	-184(%rbp), %rax
	movq	%rax, _TIG_IZ_ro2U_envp(%rip)
	nop
	movq	$1, -120(%rbp)
.L131:
	cmpq	$31, -120(%rbp)
	ja	.L134
	movq	-120(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L92(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L92(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L92:
	.long	.L134-.L92
	.long	.L115-.L92
	.long	.L134-.L92
	.long	.L114-.L92
	.long	.L113-.L92
	.long	.L112-.L92
	.long	.L134-.L92
	.long	.L134-.L92
	.long	.L111-.L92
	.long	.L110-.L92
	.long	.L109-.L92
	.long	.L108-.L92
	.long	.L107-.L92
	.long	.L106-.L92
	.long	.L105-.L92
	.long	.L104-.L92
	.long	.L103-.L92
	.long	.L102-.L92
	.long	.L134-.L92
	.long	.L134-.L92
	.long	.L101-.L92
	.long	.L100-.L92
	.long	.L134-.L92
	.long	.L99-.L92
	.long	.L98-.L92
	.long	.L97-.L92
	.long	.L96-.L92
	.long	.L134-.L92
	.long	.L95-.L92
	.long	.L94-.L92
	.long	.L93-.L92
	.long	.L91-.L92
	.text
.L97:
	cmpq	$0, -128(%rbp)
	je	.L116
	movq	$30, -120(%rbp)
	jmp	.L118
.L116:
	movq	$5, -120(%rbp)
	jmp	.L118
.L113:
	leaq	-112(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-112(%rbp), %rax
	movq	%rax, %rdi
	call	check_s3m_tracker_version
	leaq	-112(%rbp), %rax
	movq	%rax, %rdi
	call	handle_s3m_flags
	leaq	-112(%rbp), %rax
	movq	%rax, %rdi
	call	handle_stereo_toggle
	leaq	-112(%rbp), %rax
	movq	%rax, %rdi
	call	handle_s3m_channels
	movq	-136(%rbp), %rax
	movq	%rax, %rdi
	call	rewind@PLT
	movq	-136(%rbp), %rdx
	leaq	-112(%rbp), %rax
	movq	%rdx, %rcx
	movl	$96, %edx
	movl	$1, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	-136(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$20, -120(%rbp)
	jmp	.L118
.L93:
	movl	$0, -140(%rbp)
	movq	$8, -120(%rbp)
	jmp	.L118
.L105:
	movl	$1, %eax
	jmp	.L132
.L104:
	cmpl	$95, -148(%rbp)
	jbe	.L120
	movq	$12, -120(%rbp)
	jmp	.L118
.L120:
	movq	$10, -120(%rbp)
	jmp	.L118
.L91:
	movl	$2, %eax
	jmp	.L132
.L107:
	movq	$0, -136(%rbp)
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$23, -120(%rbp)
	jmp	.L118
.L111:
	leaq	-112(%rbp), %rax
	movq	%rax, %rdi
	call	check_s3m_header
	movl	%eax, -144(%rbp)
	movq	$24, -120(%rbp)
	jmp	.L118
.L115:
	movq	$16, -120(%rbp)
	jmp	.L118
.L99:
	cmpl	$1, -164(%rbp)
	jg	.L122
	cmpl	$0, -164(%rbp)
	jns	.L123
	jmp	.L124
.L122:
	cmpl	$2, -164(%rbp)
	je	.L125
	jmp	.L124
.L123:
	movq	$26, -120(%rbp)
	jmp	.L126
.L125:
	movq	$11, -120(%rbp)
	jmp	.L126
.L124:
	movq	$9, -120(%rbp)
	nop
.L126:
	jmp	.L118
.L114:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$14, -120(%rbp)
	jmp	.L118
.L103:
	movb	$0, -112(%rbp)
	movl	$1, -148(%rbp)
	movq	$15, -120(%rbp)
	jmp	.L118
.L98:
	cmpl	$0, -144(%rbp)
	je	.L127
	movq	$13, -120(%rbp)
	jmp	.L118
.L127:
	movq	$4, -120(%rbp)
	jmp	.L118
.L100:
	movl	$1, %eax
	jmp	.L132
.L96:
	movq	-176(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rsi
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$17, -120(%rbp)
	jmp	.L118
.L108:
	movq	-176(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	leaq	.LC9(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -136(%rbp)
	movq	$28, -120(%rbp)
	jmp	.L118
.L110:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$21, -120(%rbp)
	jmp	.L118
.L106:
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$31, -120(%rbp)
	jmp	.L118
.L102:
	movl	$1, %eax
	jmp	.L132
.L95:
	cmpq	$0, -136(%rbp)
	jne	.L129
	movq	$3, -120(%rbp)
	jmp	.L118
.L129:
	movq	$29, -120(%rbp)
	jmp	.L118
.L112:
	movl	$1, -140(%rbp)
	movq	$8, -120(%rbp)
	jmp	.L118
.L109:
	movl	-148(%rbp), %eax
	movb	$0, -112(%rbp,%rax)
	addl	$1, -148(%rbp)
	movq	$15, -120(%rbp)
	jmp	.L118
.L94:
	movq	-136(%rbp), %rdx
	leaq	-112(%rbp), %rax
	movq	%rdx, %rcx
	movl	$96, %edx
	movl	$1, %esi
	movq	%rax, %rdi
	call	fread@PLT
	movq	%rax, -128(%rbp)
	movq	$25, -120(%rbp)
	jmp	.L118
.L101:
	movl	$0, %eax
	jmp	.L132
.L134:
	nop
.L118:
	jmp	.L131
.L132:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L133
	call	__stack_chk_fail@PLT
.L133:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	main, .-main
	.section	.rodata
.LC12:
	.string	"%3u"
	.align 8
.LC13:
	.string	"Channel values (decimal):\n0-7: Left 1 - 8\n8-15: Right 1 - 8\n16-24: Adlib Melody 1 - 9\n25-29: Adlib Percussion (unused)\n30-127: Invalid/Garbage\nall values above + 128 = disabled\n255: Unused channel"
	.align 8
.LC14:
	.string	"Enter the value for channel %02d (decimal):"
	.text
	.globl	handle_s3m_channels
	.type	handle_s3m_channels, @function
handle_s3m_channels:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movq	%rdi, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$15, -16(%rbp)
.L163:
	cmpq	$18, -16(%rbp)
	ja	.L166
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L138(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L138(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L138:
	.long	.L166-.L138
	.long	.L166-.L138
	.long	.L167-.L138
	.long	.L151-.L138
	.long	.L150-.L138
	.long	.L149-.L138
	.long	.L166-.L138
	.long	.L148-.L138
	.long	.L147-.L138
	.long	.L146-.L138
	.long	.L145-.L138
	.long	.L167-.L138
	.long	.L143-.L138
	.long	.L142-.L138
	.long	.L166-.L138
	.long	.L141-.L138
	.long	.L140-.L138
	.long	.L139-.L138
	.long	.L137-.L138
	.text
.L137:
	movl	$1, -28(%rbp)
	movq	$16, -16(%rbp)
	jmp	.L153
.L150:
	cmpq	$0, -56(%rbp)
	jne	.L154
	movq	$2, -16(%rbp)
	jmp	.L153
.L154:
	movq	$17, -16(%rbp)
	jmp	.L153
.L141:
	movq	$13, -16(%rbp)
	jmp	.L153
.L143:
	cmpl	$0, -36(%rbp)
	je	.L156
	movq	$8, -16(%rbp)
	jmp	.L153
.L156:
	movq	$18, -16(%rbp)
	jmp	.L153
.L147:
	movl	$0, -28(%rbp)
	movq	$16, -16(%rbp)
	jmp	.L153
.L151:
	cmpq	$31, -24(%rbp)
	ja	.L158
	movq	$7, -16(%rbp)
	jmp	.L153
.L158:
	movq	$11, -16(%rbp)
	jmp	.L153
.L140:
	leaq	-40(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	%eax, -32(%rbp)
	movq	$10, -16(%rbp)
	jmp	.L153
.L146:
	movl	-40(%rbp), %edx
	movq	-24(%rbp), %rax
	leaq	64(%rax), %rcx
	movq	-56(%rbp), %rax
	addq	%rcx, %rax
	movb	%dl, (%rax)
	movq	$5, -16(%rbp)
	jmp	.L153
.L142:
	movq	$0, -24(%rbp)
	movl	$0, -40(%rbp)
	movq	$4, -16(%rbp)
	jmp	.L153
.L139:
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -24(%rbp)
	movq	$3, -16(%rbp)
	jmp	.L153
.L149:
	addq	$1, -24(%rbp)
	movq	$3, -16(%rbp)
	jmp	.L153
.L145:
	cmpl	$1, -32(%rbp)
	jne	.L161
	movq	$9, -16(%rbp)
	jmp	.L153
.L161:
	movq	$5, -16(%rbp)
	jmp	.L153
.L148:
	movq	-24(%rbp), %rax
	movzbl	%al, %eax
	addl	$1, %eax
	movl	%eax, %esi
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	%eax, -36(%rbp)
	movq	$12, -16(%rbp)
	jmp	.L153
.L166:
	nop
.L153:
	jmp	.L163
.L167:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L165
	call	__stack_chk_fail@PLT
.L165:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	handle_s3m_channels, .-handle_s3m_channels
	.section	.rodata
.LC15:
	.string	"Imago Orpheus %1X.%02X\n"
.LC16:
	.string	"Graoumf Tracker"
	.align 8
.LC17:
	.string	"Tracker info: %04X, which translates to...\n"
.LC18:
	.string	"Scream Tracker 3.%02X\n"
.LC19:
	.string	"(could be disguised...)"
.LC20:
	.string	"BeRo Tracker %1X.%02X\n"
.LC21:
	.string	"CreamTracker %1X.%02X\n"
.LC22:
	.string	"Unknown"
.LC23:
	.string	"OpenMPT %1X.%02X.%1X.%1X\n"
.LC24:
	.string	"Camoto / libgamemusic"
	.align 8
.LC25:
	.string	"Polish localized Squeak Tracker"
.LC26:
	.string	"OpenMPT %1X.%02X\n"
.LC27:
	.string	"Impulse Tracker %1X.%02X\n"
.LC28:
	.string	"Schism Tracker %1X.%02X\n"
	.text
	.globl	check_s3m_tracker_version
	.type	check_s3m_tracker_version, @function
check_s3m_tracker_version:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$144, %rsp
	movq	%rdi, -136(%rbp)
	movq	$74, -8(%rbp)
.L289:
	cmpq	$75, -8(%rbp)
	ja	.L290
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L171(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L171(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L171:
	.long	.L236-.L171
	.long	.L235-.L171
	.long	.L234-.L171
	.long	.L233-.L171
	.long	.L232-.L171
	.long	.L231-.L171
	.long	.L230-.L171
	.long	.L229-.L171
	.long	.L228-.L171
	.long	.L227-.L171
	.long	.L226-.L171
	.long	.L291-.L171
	.long	.L224-.L171
	.long	.L223-.L171
	.long	.L222-.L171
	.long	.L221-.L171
	.long	.L220-.L171
	.long	.L219-.L171
	.long	.L218-.L171
	.long	.L290-.L171
	.long	.L217-.L171
	.long	.L216-.L171
	.long	.L215-.L171
	.long	.L214-.L171
	.long	.L213-.L171
	.long	.L212-.L171
	.long	.L211-.L171
	.long	.L210-.L171
	.long	.L209-.L171
	.long	.L208-.L171
	.long	.L207-.L171
	.long	.L206-.L171
	.long	.L205-.L171
	.long	.L290-.L171
	.long	.L204-.L171
	.long	.L290-.L171
	.long	.L203-.L171
	.long	.L202-.L171
	.long	.L291-.L171
	.long	.L290-.L171
	.long	.L200-.L171
	.long	.L199-.L171
	.long	.L198-.L171
	.long	.L290-.L171
	.long	.L197-.L171
	.long	.L290-.L171
	.long	.L196-.L171
	.long	.L195-.L171
	.long	.L194-.L171
	.long	.L193-.L171
	.long	.L192-.L171
	.long	.L290-.L171
	.long	.L191-.L171
	.long	.L190-.L171
	.long	.L290-.L171
	.long	.L189-.L171
	.long	.L188-.L171
	.long	.L187-.L171
	.long	.L186-.L171
	.long	.L185-.L171
	.long	.L290-.L171
	.long	.L184-.L171
	.long	.L183-.L171
	.long	.L182-.L171
	.long	.L181-.L171
	.long	.L290-.L171
	.long	.L180-.L171
	.long	.L179-.L171
	.long	.L178-.L171
	.long	.L177-.L171
	.long	.L176-.L171
	.long	.L175-.L171
	.long	.L174-.L171
	.long	.L173-.L171
	.long	.L172-.L171
	.long	.L170-.L171
	.text
.L218:
	movl	$0, -24(%rbp)
	movq	$21, -8(%rbp)
	jmp	.L237
.L192:
	movl	$0, -48(%rbp)
	movq	$38, -8(%rbp)
	jmp	.L237
.L212:
	movl	$0, -16(%rbp)
	movq	$38, -8(%rbp)
	jmp	.L237
.L193:
	movl	$1, -44(%rbp)
	movq	$38, -8(%rbp)
	jmp	.L237
.L191:
	movq	-136(%rbp), %rax
	addq	$41, %rax
	movzbl	(%rax), %eax
	shrb	$4, %al
	movzbl	%al, %eax
	cmpl	$7, %eax
	ja	.L238
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L240(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L240(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L240:
	.long	.L238-.L240
	.long	.L246-.L240
	.long	.L245-.L240
	.long	.L244-.L240
	.long	.L243-.L240
	.long	.L242-.L240
	.long	.L241-.L240
	.long	.L239-.L240
	.text
.L239:
	movq	$57, -8(%rbp)
	jmp	.L247
.L241:
	movq	$36, -8(%rbp)
	jmp	.L247
.L242:
	movq	$75, -8(%rbp)
	jmp	.L247
.L243:
	movq	$7, -8(%rbp)
	jmp	.L247
.L244:
	movq	$64, -8(%rbp)
	jmp	.L247
.L245:
	movq	$4, -8(%rbp)
	jmp	.L247
.L246:
	movq	$1, -8(%rbp)
	jmp	.L247
.L238:
	movq	$28, -8(%rbp)
	nop
.L247:
	jmp	.L237
.L232:
	movq	-136(%rbp), %rax
	addq	$40, %rax
	movzbl	(%rax), %eax
	movzbl	%al, %eax
	movq	-136(%rbp), %rdx
	addq	$41, %rdx
	movzbl	(%rdx), %edx
	movzbl	%dl, %edx
	movl	%edx, %ecx
	andl	$15, %ecx
	movl	%eax, %edx
	movl	%ecx, %esi
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	%eax, -92(%rbp)
	movq	$20, -8(%rbp)
	jmp	.L237
.L207:
	movl	$1, -20(%rbp)
	movq	$38, -8(%rbp)
	jmp	.L237
.L183:
	movl	$0, -64(%rbp)
	movq	$38, -8(%rbp)
	jmp	.L237
.L222:
	cmpl	$0, -100(%rbp)
	je	.L248
	movq	$18, -8(%rbp)
	jmp	.L237
.L248:
	movq	$40, -8(%rbp)
	jmp	.L237
.L221:
	movl	$0, -12(%rbp)
	movq	$38, -8(%rbp)
	jmp	.L237
.L188:
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	%eax, -108(%rbp)
	movq	$27, -8(%rbp)
	jmp	.L237
.L206:
	movq	-136(%rbp), %rax
	addq	$41, %rax
	movzbl	(%rax), %eax
	movzbl	%al, %eax
	sall	$8, %eax
	movl	%eax, %edx
	movq	-136(%rbp), %rax
	addq	$40, %rax
	movzbl	(%rax), %eax
	movzbl	%al, %eax
	orl	%edx, %eax
	movw	%ax, -122(%rbp)
	movzwl	-122(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	%eax, -120(%rbp)
	movq	$22, -8(%rbp)
	jmp	.L237
.L224:
	movl	$0, -52(%rbp)
	movq	$38, -8(%rbp)
	jmp	.L237
.L177:
	movl	$1, -60(%rbp)
	movq	$38, -8(%rbp)
	jmp	.L237
.L228:
	movl	$1, -12(%rbp)
	movq	$38, -8(%rbp)
	jmp	.L237
.L235:
	movq	-136(%rbp), %rax
	addq	$40, %rax
	movzbl	(%rax), %eax
	movzbl	%al, %eax
	movl	%eax, %esi
	leaq	.LC18(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	%eax, -100(%rbp)
	movq	$14, -8(%rbp)
	jmp	.L237
.L214:
	movl	$1, -40(%rbp)
	movq	$38, -8(%rbp)
	jmp	.L237
.L176:
	cmpl	$0, -68(%rbp)
	je	.L250
	movq	$12, -8(%rbp)
	jmp	.L237
.L250:
	movq	$55, -8(%rbp)
	jmp	.L237
.L233:
	movq	-136(%rbp), %rax
	addq	$55, %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	jne	.L252
	movq	$6, -8(%rbp)
	jmp	.L237
.L252:
	movq	$63, -8(%rbp)
	jmp	.L237
.L220:
	movl	$1, -56(%rbp)
	movq	$38, -8(%rbp)
	jmp	.L237
.L213:
	movl	$1, -16(%rbp)
	movq	$38, -8(%rbp)
	jmp	.L237
.L216:
	leaq	.LC19(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	%eax, -96(%rbp)
	movq	$58, -8(%rbp)
	jmp	.L237
.L203:
	movq	-136(%rbp), %rax
	addq	$40, %rax
	movzbl	(%rax), %eax
	movzbl	%al, %eax
	movq	-136(%rbp), %rdx
	addq	$41, %rdx
	movzbl	(%rdx), %edx
	movzbl	%dl, %edx
	movl	%edx, %ecx
	andl	$15, %ecx
	movl	%eax, %edx
	movl	%ecx, %esi
	leaq	.LC20(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	%eax, -72(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L237
.L187:
	movq	-136(%rbp), %rax
	addq	$40, %rax
	movzbl	(%rax), %eax
	movzbl	%al, %eax
	movq	-136(%rbp), %rdx
	addq	$41, %rdx
	movzbl	(%rdx), %edx
	movzbl	%dl, %edx
	movl	%edx, %ecx
	andl	$15, %ecx
	movl	%eax, %edx
	movl	%ecx, %esi
	leaq	.LC21(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	%eax, -68(%rbp)
	movq	$70, -8(%rbp)
	jmp	.L237
.L178:
	movl	$0, -56(%rbp)
	movq	$38, -8(%rbp)
	jmp	.L237
.L211:
	cmpq	$0, -136(%rbp)
	jne	.L254
	movq	$11, -8(%rbp)
	jmp	.L237
.L254:
	movq	$31, -8(%rbp)
	jmp	.L237
.L227:
	movl	$1, -48(%rbp)
	movq	$38, -8(%rbp)
	jmp	.L237
.L223:
	leaq	.LC22(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	%eax, -104(%rbp)
	movq	$66, -8(%rbp)
	jmp	.L237
.L182:
	movq	-136(%rbp), %rax
	addq	$55, %rax
	movzbl	(%rax), %eax
	movzbl	%al, %edi
	movq	-136(%rbp), %rax
	addq	$54, %rax
	movzbl	(%rax), %eax
	movzbl	%al, %edx
	movq	-136(%rbp), %rax
	addq	$40, %rax
	movzbl	(%rax), %eax
	movzbl	%al, %eax
	movq	-136(%rbp), %rcx
	addq	$41, %rcx
	movzbl	(%rcx), %ecx
	movzbl	%cl, %ecx
	movl	%ecx, %esi
	andl	$15, %esi
	movl	%edi, %r8d
	movl	%edx, %ecx
	movl	%eax, %edx
	leaq	.LC23(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	%eax, -76(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L237
.L205:
	cmpw	$520, -122(%rbp)
	jne	.L257
	movq	$59, -8(%rbp)
	jmp	.L237
.L257:
	movq	$71, -8(%rbp)
	jmp	.L237
.L219:
	cmpl	$0, -112(%rbp)
	je	.L259
	movq	$29, -8(%rbp)
	jmp	.L237
.L259:
	movq	$49, -8(%rbp)
	jmp	.L237
.L200:
	movl	$1, -24(%rbp)
	movq	$21, -8(%rbp)
	jmp	.L237
.L179:
	leaq	.LC24(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	%eax, -116(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L237
.L189:
	movl	$1, -52(%rbp)
	movq	$38, -8(%rbp)
	jmp	.L237
.L185:
	leaq	.LC25(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	%eax, -112(%rbp)
	movq	$17, -8(%rbp)
	jmp	.L237
.L230:
	movq	-136(%rbp), %rax
	addq	$40, %rax
	movzbl	(%rax), %eax
	movzbl	%al, %eax
	movq	-136(%rbp), %rdx
	addq	$41, %rdx
	movzbl	(%rdx), %edx
	movzbl	%dl, %edx
	movl	%edx, %ecx
	andl	$15, %ecx
	movl	%eax, %edx
	movl	%ecx, %esi
	leaq	.LC26(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	%eax, -80(%rbp)
	movq	$41, -8(%rbp)
	jmp	.L237
.L210:
	cmpl	$0, -108(%rbp)
	je	.L261
	movq	$34, -8(%rbp)
	jmp	.L237
.L261:
	movq	$30, -8(%rbp)
	jmp	.L237
.L184:
	cmpl	$0, -88(%rbp)
	je	.L263
	movq	$15, -8(%rbp)
	jmp	.L237
.L263:
	movq	$8, -8(%rbp)
	jmp	.L237
.L186:
	cmpl	$0, -96(%rbp)
	je	.L265
	movq	$37, -8(%rbp)
	jmp	.L237
.L265:
	movq	$42, -8(%rbp)
	jmp	.L237
.L204:
	movl	$0, -20(%rbp)
	movq	$38, -8(%rbp)
	jmp	.L237
.L172:
	movw	$0, -122(%rbp)
	movq	$26, -8(%rbp)
	jmp	.L237
.L170:
	movq	-136(%rbp), %rax
	addq	$54, %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	jne	.L267
	movq	$3, -8(%rbp)
	jmp	.L237
.L267:
	movq	$63, -8(%rbp)
	jmp	.L237
.L194:
	movl	$0, -40(%rbp)
	movq	$38, -8(%rbp)
	jmp	.L237
.L175:
	cmpw	$21575, -122(%rbp)
	jne	.L269
	movq	$56, -8(%rbp)
	jmp	.L237
.L269:
	movq	$13, -8(%rbp)
	jmp	.L237
.L215:
	cmpl	$0, -120(%rbp)
	je	.L271
	movq	$47, -8(%rbp)
	jmp	.L237
.L271:
	movq	$46, -8(%rbp)
	jmp	.L237
.L209:
	cmpw	$-13824, -122(%rbp)
	jne	.L273
	movq	$67, -8(%rbp)
	jmp	.L237
.L273:
	movq	$32, -8(%rbp)
	jmp	.L237
.L190:
	movl	$0, -60(%rbp)
	movq	$38, -8(%rbp)
	jmp	.L237
.L195:
	movl	$0, -36(%rbp)
	movq	$52, -8(%rbp)
	jmp	.L237
.L173:
	movl	$1, -64(%rbp)
	movq	$38, -8(%rbp)
	jmp	.L237
.L197:
	movl	$1, -32(%rbp)
	movq	$38, -8(%rbp)
	jmp	.L237
.L231:
	cmpl	$0, -76(%rbp)
	je	.L275
	movq	$53, -8(%rbp)
	jmp	.L237
.L275:
	movq	$69, -8(%rbp)
	jmp	.L237
.L174:
	cmpl	$0, -84(%rbp)
	je	.L277
	movq	$25, -8(%rbp)
	jmp	.L237
.L277:
	movq	$24, -8(%rbp)
	jmp	.L237
.L202:
	movl	$0, -28(%rbp)
	movq	$38, -8(%rbp)
	jmp	.L237
.L181:
	movq	-136(%rbp), %rax
	addq	$40, %rax
	movzbl	(%rax), %eax
	movzbl	%al, %eax
	movq	-136(%rbp), %rdx
	addq	$41, %rdx
	movzbl	(%rdx), %edx
	movzbl	%dl, %edx
	movl	%edx, %ecx
	andl	$15, %ecx
	movl	%eax, %edx
	movl	%ecx, %esi
	leaq	.LC27(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	%eax, -88(%rbp)
	movq	$61, -8(%rbp)
	jmp	.L237
.L199:
	cmpl	$0, -80(%rbp)
	je	.L279
	movq	$62, -8(%rbp)
	jmp	.L237
.L279:
	movq	$73, -8(%rbp)
	jmp	.L237
.L226:
	movl	$0, -32(%rbp)
	movq	$38, -8(%rbp)
	jmp	.L237
.L198:
	movl	$1, -28(%rbp)
	movq	$38, -8(%rbp)
	jmp	.L237
.L236:
	cmpl	$0, -116(%rbp)
	je	.L281
	movq	$48, -8(%rbp)
	jmp	.L237
.L281:
	movq	$23, -8(%rbp)
	jmp	.L237
.L196:
	movl	$1, -36(%rbp)
	movq	$52, -8(%rbp)
	jmp	.L237
.L180:
	cmpl	$0, -104(%rbp)
	je	.L283
	movq	$68, -8(%rbp)
	jmp	.L237
.L283:
	movq	$16, -8(%rbp)
	jmp	.L237
.L229:
	movq	-136(%rbp), %rax
	addq	$40, %rax
	movzbl	(%rax), %eax
	movzbl	%al, %eax
	movq	-136(%rbp), %rdx
	addq	$41, %rdx
	movzbl	(%rdx), %edx
	movzbl	%dl, %edx
	movl	%edx, %ecx
	andl	$15, %ecx
	movl	%eax, %edx
	movl	%ecx, %esi
	leaq	.LC28(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	%eax, -84(%rbp)
	movq	$72, -8(%rbp)
	jmp	.L237
.L208:
	movl	$0, -44(%rbp)
	movq	$38, -8(%rbp)
	jmp	.L237
.L234:
	cmpl	$0, -72(%rbp)
	je	.L285
	movq	$10, -8(%rbp)
	jmp	.L237
.L285:
	movq	$44, -8(%rbp)
	jmp	.L237
.L217:
	cmpl	$0, -92(%rbp)
	je	.L287
	movq	$50, -8(%rbp)
	jmp	.L237
.L287:
	movq	$9, -8(%rbp)
	jmp	.L237
.L290:
	nop
.L237:
	jmp	.L289
.L291:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	check_s3m_tracker_version, .-check_s3m_tracker_version
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
